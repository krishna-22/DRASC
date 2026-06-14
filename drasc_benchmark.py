import argparse, os, sys, time, json, platform, warnings, pickle, urllib.request, zipfile, gc

_THREADS = str(min(8, os.cpu_count() or 8))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import (KMeans, SpectralClustering, AgglomerativeClustering,
                             DBSCAN, Birch)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score as ARI,
                             normalized_mutual_info_score as NMI,
                             silhouette_score)
from sklearn.datasets import (load_wine, load_breast_cancer, load_digits,
                              fetch_olivetti_faces, fetch_openml)

OUTDIR = "benchmark_outputs"
RESULTS = os.path.join(OUTDIR, "all_results.csv")
CACHE = os.path.join(OUTDIR, "datasets.pkl")
FIGDIR = os.path.join(OUTDIR, "figures")
GRID_K = (7, 10, 15, 30, 50)
GRID_G = (1.0, 2.0, 3.0)
SEL_SEED = 0
STOCH = {"DRASC (proposed)", "Ablation-B (adaptive only, g=0)", "k-means++",
         "GMM (EM)", "Spectral (kNN)", "Spectral (RBF)", "LSC (anchor spectral)"}


def seeds_for(n):
    if n <= 3000:
        return [0, 1, 2, 3, 4]
    if n <= 8000:
        return [0, 1, 2]
    return [0, 1]


def msd(vals):
    if not vals:
        return -1.0, 0.0
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std())


def sil(Xs, lab, rng=0):
    u = set(lab)
    if len(u) < 2 or len(u) >= len(lab):
        return -1.0
    ss = 4000 if len(lab) > 4000 else None
    try:
        return float(silhouette_score(Xs, lab, sample_size=ss, random_state=rng))
    except Exception:
        return -1.0


class DRASC:
    def __init__(self, n_clusters, n_neighbors=None, gamma=2.0, random_state=0):
        self.c = n_clusters
        self.k = n_neighbors
        self.gamma = gamma
        self.rs = random_state
        self.eigengap_ = 0.0

    def fit_predict(self, X):
        n = X.shape[0]
        k = self.k or max(10, int(np.ceil(np.log2(n) * 2)))
        k = min(k, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        dist, idx = nn.kneighbors(X)
        dist, idx = dist[:, 1:], idx[:, 1:]
        sigma = dist[:, -1].copy()
        sigma[sigma <= 0] = np.min(sigma[sigma > 0]) if np.any(sigma > 0) else 1e-12
        rho = 1.0 / sigma
        rows = np.repeat(np.arange(n), k)
        cols = idx.ravel()
        w = np.exp(-(dist ** 2).ravel() / (sigma[rows] * sigma[cols]))
        r = np.minimum(rho[rows], rho[cols]) / np.maximum(rho[rows], rho[cols])
        w = w * (r ** self.gamma)
        W = csr_matrix((w, (rows, cols)), shape=(n, n))
        W = W.maximum(W.T)
        deg = np.asarray(W.sum(axis=1)).ravel()
        deg[deg <= 0] = 1e-12
        Dh = diags(1.0 / np.sqrt(deg))
        M = Dh @ W @ Dh
        kk = min(self.c + 1, n - 2)
        vals, vecs = eigsh(M, k=kk, which='LA')
        order = np.argsort(-vals)
        lap = 1.0 - vals[order]
        self.eigengap_ = float((lap[self.c] - lap[self.c - 1]) / max(lap[self.c], 1e-12)) if kk > self.c else 0.0
        U = vecs[:, order[:self.c]]
        nrm = np.linalg.norm(U, axis=1, keepdims=True)
        nrm[nrm == 0] = 1e-12
        self.embedding_ = U / nrm
        return KMeans(self.c, n_init=30, random_state=self.rs).fit_predict(self.embedding_)


def dpc(X, n_clusters, dc_percent=2.0):
    n = X.shape[0]
    D = squareform(pdist(X))
    dc = np.percentile(D[np.triu_indices(n, 1)], dc_percent)
    rho = np.exp(-(D / dc) ** 2).sum(axis=1) - 1.0
    order = np.argsort(-rho)
    delta = np.zeros(n)
    nneigh = np.full(n, -1)
    delta[order[0]] = D[order[0]].max()
    for pos in range(1, n):
        i = order[pos]
        higher = order[:pos]
        j = higher[np.argmin(D[i, higher])]
        delta[i] = D[i, j]
        nneigh[i] = j
    centers = np.argsort(-(rho * delta))[:n_clusters]
    labels = np.full(n, -1)
    labels[centers] = np.arange(n_clusters)
    for i in order:
        if labels[i] == -1:
            labels[i] = labels[nneigh[i]]
    return labels


def lsc(X, n_clusters, n_anchors=512, k=5, random_state=0):
    n = X.shape[0]
    m = min(n_anchors, max(n // 2, n_clusters * 2))
    A = KMeans(m, n_init=3, random_state=random_state).fit(X).cluster_centers_
    dist, idx = NearestNeighbors(n_neighbors=min(k, m), n_jobs=-1).fit(A).kneighbors(X)
    sigma = dist[:, -1:].clip(1e-12)
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    Z = np.zeros((n, m))
    np.put_along_axis(Z, idx, w, axis=1)
    Z /= Z.sum(axis=1, keepdims=True).clip(1e-12)
    Zb = Z / np.sqrt(Z.sum(axis=0).clip(1e-12))
    U, _, _ = np.linalg.svd(Zb, full_matrices=False)
    E = U[:, :n_clusters]
    E /= np.linalg.norm(E, axis=1, keepdims=True).clip(1e-12)
    return KMeans(n_clusters, n_init=10, random_state=random_state).fit_predict(E)


def _sub(X, y, rng, n_max=4000):
    if len(y) <= n_max:
        return X, y
    s = rng.choice(len(y), n_max, replace=False)
    return X[s], y[s]


def get_datasets(include_images=True):
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    rng = np.random.RandomState(0)
    ds = {}
    d = load_wine()
    ds["Wine (UCI)"] = (d.data.astype(float), d.target)
    d = load_breast_cancer()
    ds["Breast Cancer Wisconsin (UCI)"] = (d.data.astype(float), d.target)
    d = load_digits()
    ds["Digits 8x8 (UCI handwritten)"] = (d.data.astype(float), d.target)
    f = fetch_olivetti_faces()
    Xf = StandardScaler().fit_transform(f.data.astype(float))
    ds["Olivetti Faces (AT&T, PCA-50)"] = (PCA(50, random_state=0).fit_transform(Xf), f.target)
    for name, key in [("Banknote Authentication (UCI)", "banknote-authentication"),
                      ("Seeds (UCI)", "seeds"), ("Pen Digits (UCI)", "pendigits"),
                      ("Satimage / Landsat (UCI)", "satimage"),
                      ("Vehicle Silhouettes (UCI)", "vehicle"),
                      ("Image Segmentation (UCI)", "segment")]:
        try:
            o = fetch_openml(key, version=1, as_frame=False, parser="auto")
            X = o.data.astype(float)
            y = pd.factorize(o.target)[0]
            X = X[:, np.nanstd(X, axis=0) > 0]
            X = np.nan_to_num(X)
            ds[name] = _sub(X, y, rng)
            print("cached:", name, ds[name][0].shape)
        except Exception as e:
            print("skip", name, repr(e)[:80])
    if include_images:
        _add_images(ds)
    pickle.dump(ds, open(CACHE, "wb"))
    return ds


def _add_images(ds):
    rng = np.random.RandomState(0)
    try:
        try:
            o = fetch_openml("USPS", as_frame=False, parser="auto")
        except Exception:
            o = fetch_openml(data_id=41070, as_frame=False, parser="auto")
        X = o.data.astype(float)
        y = pd.factorize(o.target)[0]
        Xs = StandardScaler().fit_transform(X)
        Xr = PCA(50, random_state=0).fit_transform(Xs)
        ds["USPS (PCA-50)"] = (Xr, y)
        print("cached: USPS", Xr.shape)
    except Exception as e:
        print("skip USPS", repr(e)[:80])
    try:
        if not os.path.exists(os.path.join(OUTDIR, "mnist.npz")):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
                os.path.join(OUTDIR, "mnist.npz"))
        d = np.load(os.path.join(OUTDIR, "mnist.npz"))
        X = np.concatenate([d["x_train"], d["x_test"]]).reshape(70000, -1).astype(np.float32) / 255.0
        y = np.concatenate([d["y_train"], d["y_test"]])
        idx = rng.choice(70000, 10000, replace=False)
        Xs = StandardScaler().fit_transform(X[idx])
        Xr = PCA(50, random_state=0).fit_transform(Xs)
        ds["MNIST (10k, PCA-50)"] = (Xr, y[idx])
        print("cached: MNIST", Xr.shape)
    except Exception as e:
        print("skip MNIST", repr(e)[:80])
    try:
        o = fetch_openml("Fashion-MNIST", as_frame=False, parser="auto")
        X = o.data.astype(np.float32) / 255.0
        y = pd.factorize(o.target)[0]
        idx = rng.choice(len(y), 10000, replace=False)
        Xs = StandardScaler().fit_transform(X[idx])
        Xr = PCA(50, random_state=0).fit_transform(Xs)
        ds["Fashion-MNIST (10k, PCA-50)"] = (Xr, y[idx])
        print("cached: Fashion-MNIST", Xr.shape)
    except Exception as e:
        print("skip Fashion-MNIST", repr(e)[:80])


def build_grids(Xs, c, n, p, eps_grid, anchor_grid, have_hdb, have_finch, fast):
    g = {}
    g["k-means++"] = [({"n_init": 50}, lambda s: KMeans(c, n_init=50, random_state=s).fit_predict(Xs))]
    covs = ["spherical", "diag"]
    if p * p * 8 < 3e8:
        covs.append("tied")
    if c * p * p * 8 < 1.5e9:
        covs.append("full")
    g["GMM (EM)"] = [({"cov": cv}, (lambda s, cv=cv: GaussianMixture(
        c, covariance_type=cv, n_init=5, random_state=s).fit_predict(Xs)))
        for cv in covs]
    if n <= 7000:
        g["Agglomerative"] = [({"linkage": l}, (lambda s, l=l: AgglomerativeClustering(
            c, linkage=l).fit_predict(Xs))) for l in ("ward", "complete", "average", "single")]
    else:
        conn = kneighbors_graph(Xs, 15, include_self=False, n_jobs=-1)
        g["Agglomerative"] = [({"linkage": l, "knn": 15}, (lambda s, l=l: AgglomerativeClustering(
            c, linkage=l, connectivity=conn).fit_predict(Xs))) for l in ("ward", "complete", "average")]
    g["Spectral (kNN)"] = [({"k": k}, (lambda s, k=k: SpectralClustering(
        c, affinity="nearest_neighbors", n_neighbors=k, random_state=s,
        n_jobs=-1).fit_predict(Xs))) for k in GRID_K]
    if n <= 6000:
        g["Spectral (RBF)"] = [({"gamma": gm}, (lambda s, gm=gm: SpectralClustering(
            c, affinity="rbf", gamma=gm / p, random_state=s, n_jobs=-1).fit_predict(Xs)))
            for gm in (0.1, 1.0, 10.0)]
    g["DBSCAN"] = [({"eps": round(e, 3), "ms": ms}, (lambda s, e=e, ms=ms: DBSCAN(
        eps=e, min_samples=ms, n_jobs=-1).fit_predict(Xs)))
        for e in eps_grid for ms in (5, 10)]
    if have_hdb:
        import hdbscan
        g["HDBSCAN"] = [({"mcs": m, "ms": sm}, (lambda s, m=m, sm=sm: hdbscan.HDBSCAN(
            min_cluster_size=m, min_samples=sm).fit_predict(Xs)))
            for m in (5, 10, 25, 50) for sm in (None, 5, 10)]
    g["BIRCH"] = [({"thr": t}, (lambda s, t=t: Birch(n_clusters=c, threshold=t).fit_predict(Xs)))
                  for t in (0.2, 0.5, 0.8, 1.5)]
    if have_finch:
        from finch import FINCH

        def _finch(s):
            cl, num, req = FINCH(Xs, req_clust=c, verbose=False)
            return req if req is not None else cl[:, int(np.argmin(np.abs(np.array(num) - c)))]
        g["FINCH (2019)"] = [({"req_clust": c}, _finch)]
    if n <= 4000:
        g["DPC (Sci.2014)"] = [({"dc%": q}, (lambda s, q=q: dpc(Xs, c, q)))
                               for q in (0.5, 1.0, 2.0, 5.0)]
    g["LSC (anchor spectral)"] = [({"anchors": m, "k": k}, (lambda s, m=m, k=k: lsc(Xs, c, m, k, s)))
                                  for m in anchor_grid for k in (3, 5, 10) if m < n]
    if fast:
        g = {kk: vv[:2] for kk, vv in g.items() if kk in ("k-means++", "GMM (EM)", "Spectral (kNN)")}
    return g


def drasc_scan(Xs, c, n, rng_sil, fast):
    gammas = (0.0,) + GRID_G
    ks = GRID_K if not fast else (10, 15)
    gammas = gammas if not fast else (0.0, 2.0)
    out = {}
    for k in ks:
        kk = min(k, n - 1)
        for gm in gammas:
            try:
                mdl = DRASC(c, kk, gm, SEL_SEED)
                lab = mdl.fit_predict(Xs)
                out[(k, gm)] = dict(lab=lab, gap=mdl.eigengap_, sil=sil(Xs, lab, rng_sil))
            except Exception as e:
                print("  drasc fail", k, gm, repr(e)[:40])
    return out


def stability(Xs, c, k, g, n_sub=5, frac=0.8, seed=0):
    rng = np.random.RandomState(seed)
    n = len(Xs)
    labs = []
    for s in range(n_sub):
        idx = rng.choice(n, int(frac * n), replace=False)
        try:
            labs.append((idx, DRASC(c, min(k, len(idx) - 1), g, s).fit_predict(Xs[idx])))
        except Exception:
            pass
    sc = []
    for a in range(len(labs)):
        for b in range(a + 1, len(labs)):
            (ia, la), (ib, lb) = labs[a], labs[b]
            common = np.intersect1d(ia, ib)
            if len(common) > 10:
                pa = {v: i for i, v in enumerate(ia)}
                pb = {v: i for i, v in enumerate(ib)}
                sc.append(ARI([la[pa[x]] for x in common], [lb[pb[x]] for x in common]))
    return float(np.mean(sc)) if sc else -1.0


def already(done, exp, dataset, algo):
    if not len(done):
        return False
    return ((done["experiment"] == exp) & (done["dataset"] == dataset) &
            (done["algo"] == algo)).any()


def append_rows(rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False)


def row(experiment, dataset, n, p, c, algo, ari_m, ari_s, nmi_m, nmi_s, cfg, ncfg, sec, sel):
    return dict(experiment=experiment, dataset=dataset, n=n, dims=p, classes=c, algo=algo,
                ARI_mean=round(ari_m, 4), ARI_std=round(ari_s, 4),
                NMI_mean=round(nmi_m, 4), NMI_std=round(nmi_s, 4),
                selection=sel, cfg=str(cfg), n_cfg=ncfg, sec=round(sec, 2))


def eval_dataset(name, X, y, done, have_hdb, have_finch, fast):
    Xs = StandardScaler().fit_transform(X)
    n, p = Xs.shape
    c = int(len(np.unique(y)))
    seeds = [0] if fast else seeds_for(n)
    rng_sil = 0
    kd = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(Xs).kneighbors(Xs)[0]
    eps_grid = [float(np.percentile(kd[:, j], q)) for j in (4, 9) for q in (25, 50, 75, 90)]
    anchor_grid = (32, 64, 128) if n < 300 else (256, 512, 1024)
    rows = []

    dsc = drasc_scan(Xs, c, n, rng_sil, fast)
    if dsc:
        prop = {kg: v for kg, v in dsc.items() if kg[1] > 0}
        abl = {kg: v for kg, v in dsc.items() if kg[1] == 0}

        def best_by(d, key):
            return max(d.items(), key=lambda kv: (key(kv[1]),))[0] if d else None

        def seed_eval_drasc(kg):
            k, gm = kg
            aris, nmis = [], []
            for s in seeds:
                try:
                    l = DRASC(c, min(k, n - 1), gm, s).fit_predict(Xs)
                    aris.append(ARI(y, l))
                    nmis.append(NMI(y, l))
                except Exception:
                    pass
            am, asd = msd(aris)
            nm, nsd = msd(nmis)
            return am, asd, nm, nsd

        if prop:
            t0 = time.time()
            o = max(prop.items(), key=lambda kv: ARI(y, kv[1]["lab"]))[0]
            am, asd, nm, nsd = seed_eval_drasc(o)
            rows.append(row("main_benchmark", name, n, p, c, "DRASC (proposed)",
                            am, asd, nm, nsd, {"k": o[0], "gamma": o[1]}, len(prop), time.time() - t0, "oracle"))
            su = max(prop.items(), key=lambda kv: kv[1]["sil"])[0]
            am, asd, nm, nsd = seed_eval_drasc(su)
            rows.append(row("unsup_silhouette", name, n, p, c, "DRASC (proposed)",
                            am, asd, nm, nsd, {"k": su[0], "gamma": su[1]}, len(prop), time.time() - t0, "silhouette"))
            se = max(prop.items(), key=lambda kv: kv[1]["gap"])[0]
            am, asd, nm, nsd = seed_eval_drasc(se)
            rows.append(row("drasc_selection", name, n, p, c, "DRASC (proposed)",
                            am, asd, nm, nsd, {"k": se[0], "gamma": se[1]}, len(prop), 0, "eigengap"))
            rows.append(row("drasc_selection", name, n, p, c, "DRASC (proposed)",
                            ARI(y, prop[su]["lab"]), 0, NMI(y, prop[su]["lab"]), 0,
                            {"k": su[0], "gamma": su[1]}, len(prop), 0, "silhouette"))
            rows.append(row("drasc_selection", name, n, p, c, "DRASC (proposed)",
                            ARI(y, prop[o]["lab"]), 0, NMI(y, prop[o]["lab"]), 0,
                            {"k": o[0], "gamma": o[1]}, len(prop), 0, "oracle"))
            if n <= 4000 and not fast:
                ss = max(prop.keys(), key=lambda kg: stability(Xs, c, kg[0], kg[1]))
                rows.append(row("drasc_selection", name, n, p, c, "DRASC (proposed)",
                                ARI(y, prop[ss]["lab"]), 0, NMI(y, prop[ss]["lab"]), 0,
                                {"k": ss[0], "gamma": ss[1]}, len(prop), 0, "stability"))
        if abl:
            t0 = time.time()
            o = max(abl.items(), key=lambda kv: ARI(y, kv[1]["lab"]))[0]
            am, asd, nm, nsd = seed_eval_drasc(o)
            rows.append(row("main_benchmark", name, n, p, c, "Ablation-B (adaptive only, g=0)",
                            am, asd, nm, nsd, {"k": o[0], "gamma": 0.0}, len(abl), time.time() - t0, "oracle"))
            su = max(abl.items(), key=lambda kv: kv[1]["sil"])[0]
            am, asd, nm, nsd = seed_eval_drasc(su)
            rows.append(row("unsup_silhouette", name, n, p, c, "Ablation-B (adaptive only, g=0)",
                            am, asd, nm, nsd, {"k": su[0], "gamma": 0.0}, len(abl), time.time() - t0, "silhouette"))

    grids = build_grids(Xs, c, n, p, eps_grid, anchor_grid, have_hdb, have_finch, fast)
    for algo, cfgs in grids.items():
        t0 = time.time()
        entries = []
        for cfg, fn in cfgs:
            try:
                lab = fn(SEL_SEED)
                entries.append((cfg, fn, lab, ARI(y, lab), NMI(y, lab), sil(Xs, lab, rng_sil)))
            except Exception as e:
                print("  fail", algo, cfg, repr(e)[:50])
        if not entries:
            continue
        sds = seeds if algo in STOCH else [0]

        def seed_eval(fn):
            aris, nmis = [], []
            for s in sds:
                try:
                    l = fn(s)
                    aris.append(ARI(y, l))
                    nmis.append(NMI(y, l))
                except Exception:
                    pass
            return msd(aris) + msd(nmis)

        oc = max(entries, key=lambda e: e[3])
        am, asd, nm, nsd = seed_eval(oc[1])
        rows.append(row("main_benchmark", name, n, p, c, algo, am, asd, nm, nsd,
                        oc[0], len(entries), time.time() - t0, "oracle"))
        uc = max(entries, key=lambda e: e[5])
        am, asd, nm, nsd = seed_eval(uc[1])
        rows.append(row("unsup_silhouette", name, n, p, c, algo, am, asd, nm, nsd,
                        uc[0], len(entries), time.time() - t0, "silhouette"))
        gc.collect()

    rows = [r for r in rows if not already(done, r["experiment"], name, r["algo"])]
    append_rows(rows)
    print("done:", name, "n=%d p=%d c=%d" % (n, p, c), "rows=%d" % len(rows))
    return rows


def run_main(datasets, have_hdb, have_finch, fast):
    done = pd.read_csv(RESULTS) if os.path.exists(RESULTS) else pd.DataFrame(
        columns=["experiment", "dataset", "algo"])
    for name, (X, y) in datasets.items():
        eval_dataset(name, X, y, done, have_hdb, have_finch, fast)


def run_robustness(datasets, fast):
    if "Digits 8x8 (UCI handwritten)" not in datasets:
        return
    X, y = datasets["Digits 8x8 (UCI handwritten)"]
    Xs = StandardScaler().fit_transform(X)
    c = int(len(np.unique(y)))
    n, p = Xs.shape
    seeds = [0] if fast else [0, 1, 2]
    rng = np.random.RandomState(0)

    def methods(Z, yy):
        m = {"DRASC": lambda s: DRASC(c, 15, 2.0, s).fit_predict(Z),
             "Spectral-kNN": lambda s: SpectralClustering(c, affinity="nearest_neighbors",
                                                          n_neighbors=15, random_state=s, n_jobs=-1).fit_predict(Z),
             "k-means++": lambda s: KMeans(c, n_init=20, random_state=s).fit_predict(Z),
             "GMM (EM)": lambda s: GaussianMixture(c, covariance_type="full", n_init=3,
                                                   random_state=s).fit_predict(Z),
             "LSC": lambda s: lsc(Z, c, 512, 5, s)}
        return m
    rows = []
    for noise in (0.0, 0.5, 1.0, 2.0, 4.0):
        Xn = Xs + rng.normal(0, noise, Xs.shape)
        for algo, fn in methods(Xn, y).items():
            aris, nmis = [], []
            for s in seeds:
                try:
                    l = fn(s)
                    aris.append(ARI(y, l))
                    nmis.append(NMI(y, l))
                except Exception:
                    pass
            am, asd = msd(aris)
            nm, nsd = msd(nmis)
            rows.append(row("robustness_noise", "Digits", n, p, c, algo, am, asd, nm, nsd,
                            "noise_sd=%s" % noise, 1, 0, "fixed"))
    for imb in (1.0, 0.5, 0.25, 0.1):
        keep = []
        for cl in np.unique(y):
            idx = np.where(y == cl)[0]
            frac = imb if cl % 2 == 0 else 1.0
            keep.extend(rng.choice(idx, max(5, int(frac * len(idx))), replace=False))
        keep = np.array(keep)
        Xi, yi = Xs[keep], y[keep]
        for algo, fn in methods(Xi, yi).items():
            aris, nmis = [], []
            for s in seeds:
                try:
                    l = fn(s)
                    aris.append(ARI(yi, l))
                    nmis.append(NMI(yi, l))
                except Exception:
                    pass
            am, asd = msd(aris)
            nm, nsd = msd(nmis)
            rows.append(row("robustness_imbalance", "Digits", len(yi), p, c, algo, am, asd, nm, nsd,
                            "minority_frac=%s" % imb, 1, 0, "fixed"))
    append_rows(rows)
    print("robustness done", len(rows), "rows")


def run_sensitivity(datasets, fast):
    rows = []
    targets = [d for d in ("Digits 8x8 (UCI handwritten)", "Pen Digits (UCI)") if d in datasets]
    ks = (5, 7, 10, 15, 20, 30, 50)
    gs = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    if fast:
        targets = targets[:1]
        ks, gs = (10, 15), (0.0, 2.0)
    for name in targets:
        X, y = datasets[name]
        Xs = StandardScaler().fit_transform(X)
        c = int(len(np.unique(y)))
        n, p = Xs.shape
        for k in ks:
            for g in gs:
                try:
                    l = DRASC(c, min(k, n - 1), g, 0).fit_predict(Xs)
                    rows.append(row("sensitivity", name, n, p, c, "DRASC",
                                    ARI(y, l), 0, NMI(y, l), 0, "k=%d;gamma=%s" % (k, g), 1, 0, "fixed"))
                except Exception:
                    pass
        print("sensitivity done:", name)
    append_rows(rows)


def run_scalability(fast):
    rows = []
    path = os.path.join(OUTDIR, "mnist.npz")
    try:
        if not os.path.exists(path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", path)
        d = np.load(path)
        X = np.concatenate([d["x_train"], d["x_test"]]).reshape(70000, -1).astype(np.float32) / 255.0
        y = np.concatenate([d["y_train"], d["y_test"]])
    except Exception as e:
        print("scalability skipped:", repr(e)[:80])
        return
    sizes = (2000, 5000) if fast else (5000, 10000, 20000, 40000, 70000)
    for N in sizes:
        rng = np.random.RandomState(0)
        s = rng.choice(70000, N, replace=False) if N < 70000 else np.arange(70000)
        Xn, yn = X[s], y[s]
        try:
            t0 = time.time()
            Xp = PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(Xn))
            l = DRASC(10, 15, 2.0, 0).fit_predict(Xp)
            t = time.time() - t0
            a, m = ARI(yn, l), NMI(yn, l)
            print("N=%6d time=%7.1fs ARI=%.4f NMI=%.4f" % (N, t, a, m))
            rows.append(row("scalability_mnist", "MNIST (n=%d)" % N, N, 784, 10, "DRASC (proposed)",
                            a, 0, m, 0, "k=15;gamma=2.0;pca=50", 1, t, "fixed"))
        except MemoryError:
            print("MemoryError at N=%d, stopping scalability sweep" % N)
            break
        except Exception as e:
            print("scalability fail N=%d %s" % (N, repr(e)[:60]))
    append_rows(rows)


def _pivot(df, exp, metric, sel=None):
    sub = df[df.experiment == exp]
    if sel:
        sub = sub[sub.selection == sel]
    return sub.pivot_table(index="dataset", columns="algo", values=metric)


def export():
    if not os.path.exists(RESULTS):
        print("no results to export")
        return
    df = pd.read_csv(RESULTS)
    df = df.drop_duplicates(subset=["experiment", "dataset", "algo", "selection", "cfg"], keep="last")
    df.to_csv(RESULTS, index=False)
    sheets = {}
    report = []

    def _uniq(name):
        base = name[:31]
        out = base
        i = 1
        while out in sheets:
            i += 1
            out = (base[:29] + str(i))[:31]
        return out

    def add(title, table):
        sheets[_uniq(title)] = table
        report.append("\n===== %s =====\n%s\n" % (title, table.round(3).to_string()))

    for sel, tag in (("oracle", "main_benchmark"), ("silhouette", "unsup_silhouette")):
        for metric in ("ARI_mean", "NMI_mean"):
            piv = _pivot(df, tag, metric, sel)
            if piv.empty:
                continue
            add("%s %s" % (tag, metric), piv)
            ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
            sheets[_uniq("rank_%s_%s" % (tag, metric))] = ranks.to_frame("mean_rank")
            report.append("\nMean rank (%s, %s):\n%s\n" % (tag, metric, ranks.round(2).to_string()))
            wins = piv.idxmax(axis=1).value_counts()
            report.append("Wins (%s, %s):\n%s\n" % (tag, metric, wins.to_string()))
            if metric == "ARI_mean":
                full = [piv[col] for col in piv.columns if piv[col].notna().all()]
                if len(full) >= 3 and piv.shape[0] >= 3:
                    try:
                        st, pv = friedmanchisquare(*full)
                        report.append("Friedman (%s ARI): chi2=%.3f p=%.5g\n" % (tag, st, pv))
                    except Exception:
                        pass
                if "DRASC (proposed)" in piv.columns:
                    for rival in ("Spectral (kNN)", "GMM (EM)", "k-means++",
                                  "LSC (anchor spectral)", "Agglomerative", "BIRCH", "DBSCAN"):
                        if rival in piv.columns:
                            pair = piv[["DRASC (proposed)", rival]].dropna()
                            if len(pair) >= 5 and (pair["DRASC (proposed)"] - pair[rival]).abs().sum() > 0:
                                try:
                                    _, pw = wilcoxon(pair["DRASC (proposed)"], pair[rival])
                                    report.append("Wilcoxon DRASC vs %s (%s): p=%.4g\n" % (rival, tag, pw))
                                except Exception:
                                    pass

    dsel = df[df.experiment == "drasc_selection"]
    if not dsel.empty:
        piv = dsel.pivot_table(index="dataset", columns="selection", values="ARI_mean")
        cols = [c for c in ("eigengap", "silhouette", "stability", "oracle") if c in piv.columns]
        add("DRASC selection ARI", piv[cols])
        if "oracle" in piv.columns:
            gap = piv[cols].copy()
            for c in cols:
                gap[c] = piv["oracle"] - piv[c]
            sheets[_uniq("DRASC sel gap to oracle")] = gap
            report.append("\nDRASC selection gap to oracle (ARI):\n%s\n" % gap.round(3).to_string())

    for exp in ("robustness_noise", "robustness_imbalance"):
        sub = df[df.experiment == exp]
        if not sub.empty:
            piv = sub.pivot_table(index="cfg", columns="algo", values="ARI_mean")
            add(exp, piv)

    sens = df[df.experiment == "sensitivity"]
    if not sens.empty:
        for name in sens.dataset.unique():
            s = sens[sens.dataset == name].copy()
            s["k"] = s.cfg.str.extract(r"k=(\d+)").astype(int)
            s["gamma"] = s.cfg.str.extract(r"gamma=([\d.]+)").astype(float)
            piv = s.pivot_table(index="k", columns="gamma", values="ARI_mean")
            add("sensitivity %s" % name.split()[0], piv)

    sc = df[df.experiment == "scalability_mnist"]
    if not sc.empty:
        t = sc[["dataset", "n", "sec", "ARI_mean", "NMI_mean"]].sort_values("n").set_index("dataset")
        add("scalability MNIST", t)

    for sel, tag in (("oracle", "main_benchmark"), ("silhouette", "unsup_silhouette")):
        for metric in ("ARI_mean", "NMI_mean"):
            piv = _pivot(df, tag, metric, sel)
            if not piv.empty:
                piv.round(4).to_csv(os.path.join(OUTDIR, "table_%s_%s.csv" % (tag, metric)))

    try:
        with pd.ExcelWriter(os.path.join(OUTDIR, "benchmark_summary.xlsx"), engine="openpyxl") as xw:
            for title, table in sheets.items():
                table.round(4).to_excel(xw, sheet_name=title)
        print("wrote benchmark_summary.xlsx")
    except Exception as e:
        print("xlsx export skipped:", repr(e)[:80])
        for title, table in sheets.items():
            table.round(4).to_csv(os.path.join(OUTDIR, "sheet_%s.csv" % title.replace(" ", "_")))

    txt = "DRASC BENCHMARK SUMMARY\ngenerated: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S") + "".join(report)
    open(os.path.join(OUTDIR, "benchmark_report.txt"), "w").write(txt)
    print("wrote benchmark_report.txt")


def write_env(total_sec):
    info = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threads_env": os.environ.get("OMP_NUM_THREADS"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "total_runtime_sec": round(total_sec, 1),
    }
    try:
        import sklearn, scipy
        info["scikit_learn"] = sklearn.__version__
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    for opt in ("hdbscan", "finch"):
        try:
            __import__(opt)
            info[opt] = "available"
        except Exception:
            info[opt] = "missing"
    open(os.path.join(OUTDIR, "environment.json"), "w").write(json.dumps(info, indent=2))
    print("wrote environment.json")


def bundle():
    zpath = os.path.join(OUTDIR, "drasc_results_bundle.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(OUTDIR):
            if f.endswith((".csv", ".xlsx", ".txt", ".json")):
                z.write(os.path.join(OUTDIR, f), f)
        if os.path.isdir(FIGDIR):
            for f in sorted(os.listdir(FIGDIR)):
                if f.endswith((".png", ".pdf")):
                    z.write(os.path.join(FIGDIR, f), os.path.join("figures", f))
    print("wrote", zpath)


VIZ_DATASETS = ["Digits 8x8 (UCI handwritten)", "Pen Digits (UCI)", "USPS (PCA-50)",
                "MNIST (10k, PCA-50)", "Fashion-MNIST (10k, PCA-50)",
                "Olivetti Faces (AT&T, PCA-50)"]


def _drasc_cfg_for(name):
    if not os.path.exists(RESULTS):
        return 15, 2.0
    try:
        import ast
        df = pd.read_csv(RESULTS)
        sub = df[(df.dataset == name) & (df.algo == "DRASC (proposed)") &
                 (df.experiment == "main_benchmark")]
        if len(sub):
            d = ast.literal_eval(str(sub.iloc[0]["cfg"]))
            return int(d.get("k", 15)), float(d.get("gamma", 2.0))
    except Exception:
        pass
    return 15, 2.0


def _project(Z, method, seed, n_iter, umap_mod):
    Z = np.asarray(Z, float)
    if Z.shape[1] == 2:
        return Z
    if method == "umap" and umap_mod is not None:
        return umap_mod.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                             min_dist=0.1).fit_transform(Z)
    from sklearn.manifold import TSNE
    perp = float(min(30, max(5, Z.shape[0] // 100)))
    try:
        return TSNE(n_components=2, init="pca", perplexity=perp, max_iter=n_iter,
                    random_state=seed).fit_transform(Z)
    except TypeError:
        return TSNE(n_components=2, init="pca", perplexity=perp, n_iter=n_iter,
                    random_state=seed).fit_transform(Z)


def _viz_sil(Z, y):
    try:
        ss = 3000 if len(y) > 3000 else None
        return float(silhouette_score(Z, y, sample_size=ss, random_state=0))
    except Exception:
        return float("nan")


def _scatter(plt, ax, P, y, title):
    classes = np.unique(y)
    cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
    for i, cl in enumerate(classes):
        m = y == cl
        ax.scatter(P[m, 0], P[m, 1], s=6, color=cmap(i % cmap.N), alpha=0.7,
                   linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_embedding_figure(plt, umap_mod, name, X, y, method, seed, n_iter, cap, ablation):
    rng = np.random.RandomState(seed)
    Xs = StandardScaler().fit_transform(X)
    if len(y) > cap:
        idx = rng.choice(len(y), cap, replace=False)
        Xs, y = Xs[idx], y[idx]
    c = int(len(np.unique(y)))
    k, g = _drasc_cfg_for(name)
    k = min(k, len(y) - 1)
    panels = [("Standardized input (raw)", Xs)]
    if ablation:
        m0 = DRASC(c, k, 0.0, seed)
        m0.fit_predict(Xs)
        panels.append((r"DRASC embedding ($\gamma=0$, adaptive only)", m0.embedding_))
    md = DRASC(c, k, g, seed)
    md.fit_predict(Xs)
    panels.append((r"DRASC embedding ($\gamma=%.0f$, full)" % g, md.embedding_))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, (lab, Z) in zip(axes, panels):
        s = _viz_sil(Z, y)
        P = _project(Z, method, seed, n_iter, umap_mod)
        _scatter(plt, ax, P, y, "%s\nsilhouette vs true labels = %.3f" % (lab, s))
    proj = "UMAP" if (method == "umap" and umap_mod is not None) else "t-SNE"
    fig.suptitle("%s  (n=%d, classes=%d, DRASC k=%d, %s 2-D projection)"
                 % (name, len(y), c, k, proj), fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    base = os.path.join(FIGDIR, "embedding_" + "".join(
        ch if ch.isalnum() else "_" for ch in name)[:40])
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("figure:", base + ".png/.pdf")


def run_visualizations(datasets, names=None, method="tsne", ablation=True, cap=3000,
                       n_iter=1000, fast=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib missing -> visualizations skipped (pip install matplotlib):", repr(e)[:60])
        return
    try:
        import umap as umap_mod
    except Exception:
        umap_mod = None
        if method == "umap":
            print("umap-learn missing -> using t-SNE (pip install umap-learn)")
    if fast:
        cap, n_iter = 1200, 400
    if names:
        sel = [n for n in names if n in datasets]
    else:
        sel = [n for n in VIZ_DATASETS if n in datasets] or list(datasets.keys())
    for name in sel:
        X, y = datasets[name]
        try:
            make_embedding_figure(plt, umap_mod, name, X, y, method, 0, n_iter, cap, ablation)
        except Exception as e:
            print("viz fail", name, repr(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mnist", action="store_true")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-method", choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    run_everything = not any([a.all, a.mnist, a.robustness, a.sensitivity, a.viz,
                              a.summary, a.smoke])
    fast = a.smoke
    t_start = time.time()

    try:
        import hdbscan
        have_hdb = True
    except Exception:
        have_hdb = False
        print("hdbscan missing -> HDBSCAN skipped (pip install hdbscan)")
    try:
        import finch
        have_finch = True
    except Exception:
        have_finch = False
        print("finch-clust missing -> FINCH skipped (pip install finch-clust)")

    need_data = run_everything or a.all or a.robustness or a.sensitivity or a.viz or a.smoke
    if need_data:
        include_images = not (a.no_images or a.smoke)
        if fast:
            global CACHE
            datasets = {}
            rng = np.random.RandomState(0)
            d = load_wine()
            datasets["Wine (UCI)"] = (d.data.astype(float), d.target)
            d = load_breast_cancer()
            datasets["Breast Cancer Wisconsin (UCI)"] = (d.data.astype(float), d.target)
            d = load_digits()
            datasets["Digits 8x8 (UCI handwritten)"] = (d.data.astype(float), d.target)
        else:
            datasets = get_datasets(include_images=include_images)
            print("datasets:", list(datasets.keys()))

    if run_everything or a.all or a.smoke:
        run_main(datasets, have_hdb, have_finch, fast)
    if run_everything or a.robustness or a.smoke:
        run_robustness(datasets, fast)
    if run_everything or a.sensitivity or a.smoke:
        run_sensitivity(datasets, fast)
    if run_everything or a.mnist or a.smoke:
        run_scalability(fast)

    export()
    if run_everything or a.viz or a.smoke:
        run_visualizations(datasets, method=a.viz_method, ablation=not a.no_ablation, fast=fast)
    write_env(time.time() - t_start)
    bundle()
    print("\nTOTAL RUNTIME: %.1f min" % ((time.time() - t_start) / 60.0))


if __name__ == "__main__":
    main()
