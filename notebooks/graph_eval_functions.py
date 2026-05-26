"""
Graph evaluation functions for multiprocessing.
All functions needed by evaluate_article must be importable.
"""
import random
import re
import signal

import networkx as nx
import pandas as pd


def _edge_type(label: str) -> str:
    if label == 'hierarchy':
        return 'hierarchy'
    if label.startswith('direction'):
        return 'directional'
    if label.startswith('correlation'):
        return 'correlational'
    if label.startswith('moderation'):
        return 'moderation'
    return 'unknown'


def _strip_bracket_suffix(text: str) -> str:
    """Strip trailing bracket annotations that are not validation labels.
    Removes e.g. [CASE 2], [Case 1], [note], etc. but keeps [validated/null/hypothesized]."""
    # Remove any trailing [...] that is NOT a recognised validation label
    return re.sub(r'\s*\[(?!validated|null|hypothesized)[^\]]*\]\s*$', '', text.strip()).strip()


def _validation_from_text(text: str, default: str = 'hypothesized') -> str:
    t = str(text).lower()
    if 'validated' in t:
        return 'validated'
    if 'null' in t:
        return 'null'
    if 'hypothesized' in t or 'hypothesis' in t:
        return 'hypothesized'
    return default


def parse_gt_graph(article: dict) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    ann = article['annotations'][0]
    vm = {}
    for r in ann['result']:
        if r.get('type') == 'labels' and 'Variable' in r['value'].get('labels', []):
            vm[r['id']] = r['value'].get('text', '').strip()
    for r in ann['result']:
        if r.get('type') == 'textarea' and r.get('from_name') == 'var_name' and r['id'] in vm:
            nn = r['value'].get('text', '')
            if isinstance(nn, list):
                nn = nn[0] if nn else ''
            if nn.strip():
                vm[r['id']] = nn.strip()
    for rid, name in vm.items():
        G.add_node(rid, label=name)
    for r in ann['result']:
        if r.get('type') == 'relation':
            fid, tid = r.get('from_id', ''), r.get('to_id', '')
            if fid in vm and tid in vm:
                for lbl in r.get('labels', []):
                    et = _edge_type(lbl)
                    if et != 'unknown':
                        G.add_edge(
                            fid,
                            tid,
                            edge_type=et,
                            validation=_validation_from_text(lbl, default='validated'),
                            raw_label=lbl,
                        )
    return G


def _safe(val) -> str:
    return '' if pd.isna(val) else str(val).strip()


def _smart_split(text: str, delim: str = ';') -> list:
    raw_parts = text.split(delim)
    result = []
    buf = ''
    depth = 0
    for part in raw_parts:
        buf = (buf + delim + part) if buf else part
        depth += part.count('(') - part.count(')')
        if depth <= 0:
            result.append(buf.strip())
            buf = ''
            depth = 0
    if buf:
        result.append(buf.strip())
    return result


def parse_pred_graph(row: pd.Series, mode: str) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    seen = set()

    def _ensure(n):
        if n and n not in G:
            G.add_node(n, label=n)

    def _add_edge(s, t, et, validation='hypothesized'):
        key = (s, t, et, validation)
        if key not in seen:
            G.add_edge(s, t, edge_type=et, validation=validation)
            seen.add(key)

    for v in _smart_split(_safe(row.get('step2_final_vars', ''))):
        if v:
            _ensure(v)

    hier_col = 'step2_hierarchy'
    if mode == 'step4':
        dir_col, corr_col, mod_col = 'step4_directional', 'step4_correlational', 'step4_moderation'
    elif mode == 'step5_1':
        dir_col, corr_col, mod_col = 'step5_1_directional', 'step5_1_correlational', 'step5_1_moderation'
        if 'step5_1_hierarchy' in row.index:
            hier_col = 'step5_1_hierarchy'
    elif mode == 'step5_2':
        dir_col, corr_col, mod_col = 'step5_2_directional', 'step5_2_correlational', 'step5_2_moderation'
        if 'step5_2_hierarchy' in row.index:
            hier_col = 'step5_2_hierarchy'
    else:
        raise ValueError(f'Unsupported mode: {mode}')

    for e in _smart_split(_safe(row.get(hier_col, ''))):
        if '->' in e:
            p = e.split('->')
            if len(p) == 2:
                s, t = _strip_bracket_suffix(p[0]), _strip_bracket_suffix(p[1])
                if s and t:
                    _ensure(s); _ensure(t); _add_edge(s, t, 'hierarchy', 'validated')

    for e in _smart_split(_safe(row.get(dir_col, ''))):
        m = re.search(r'\[(validated|null|hypothesized)\]$', e.strip())
        validation = m.group(1) if m else 'hypothesized'
        if m:
            e = e[: m.start()].strip()
        if '->' in e:
            p = e.split('->')
            if len(p) == 2:
                s, t = _strip_bracket_suffix(p[0]), _strip_bracket_suffix(p[1])
                if s and t:
                    _ensure(s); _ensure(t); _add_edge(s, t, 'directional', validation)

    for e in _smart_split(_safe(row.get(corr_col, ''))):
        m = re.search(r'\[(validated|null|hypothesized)\]$', e.strip())
        validation = m.group(1) if m else 'hypothesized'
        if m:
            e = e[: m.start()].strip()
        if '<->' in e:
            p = e.split('<->')
            if len(p) == 2:
                s, t = _strip_bracket_suffix(p[0]), _strip_bracket_suffix(p[1])
                if s and t:
                    _ensure(s); _ensure(t); _add_edge(s, t, 'correlational', validation)

    for e in _smart_split(_safe(row.get(mod_col, ''))):
        e = e.strip()
        if not e:
            continue
        m = re.search(r'\[(validated|null|hypothesized)\]$', e)
        validation = m.group(1) if m else 'hypothesized'
        if m:
            e = e[: m.start()].strip()
        mm = re.match(r'(.+?)\s+moderates\s+\((.+)\)', e)
        if mm:
            mod = mm.group(1).strip()
            for mv in _smart_split(mm.group(2), delim=','):
                mv = mv.strip()
                if mv:
                    _ensure(mod); _ensure(mv); _add_edge(mod, mv, 'moderation', validation)

    return G


def canonicalise_corr(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph()
    seen = set()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)

    def _lbl(n):
        return G.nodes[n].get('label', str(n))

    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', '')
        if et == 'correlational':
            lu, lv = _lbl(u), _lbl(v)
            cu, cv = (v, u) if (lu, str(u)) > (lv, str(v)) else (u, v)
            k = (cu, cv, et)
            if k not in seen:
                H.add_edge(cu, cv, **dict(d))
                seen.add(k)
        else:
            k = (u, v, et)
            if k not in seen:
                H.add_edge(u, v, **dict(d))
                seen.add(k)
    return H


def propagate_to_upper(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    H = G.copy()
    while True:
        added = 0
        hier_pairs = [(u, v) for u, v, d in H.edges(data=True) if d.get('edge_type') == 'hierarchy']
        existing = {(u, v, d.get('edge_type')) for u, v, d in H.edges(data=True)}
        for upper, lower in hier_pairs:
            for _, tgt, d in list(H.out_edges(lower, data=True)):
                et = d.get('edge_type')
                if et != 'hierarchy' and (upper, tgt, et) not in existing:
                    H.add_edge(upper, tgt, **d)
                    existing.add((upper, tgt, et))
                    added += 1
            for src, _, d in list(H.in_edges(lower, data=True)):
                et = d.get('edge_type')
                if et != 'hierarchy' and (src, upper, et) not in existing:
                    H.add_edge(src, upper, **d)
                    existing.add((src, upper, et))
                    added += 1
        if added == 0:
            break
    return H


def drop_nonhierarchy_between_hierarchy_linked(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    If two nodes are connected by a hierarchy link in either direction,
    keep only hierarchy edges between that node pair.
    """
    H = nx.MultiDiGraph()
    seen = set()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)

    hier_pairs = {frozenset((u, v)) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'hierarchy' and u != v}

    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', '')
        if et != 'hierarchy' and frozenset((u, v)) in hier_pairs:
            continue
        k = (u, v, et)
        if k not in seen:
            H.add_edge(u, v, **dict(d))
            seen.add(k)
    return H


def drop_corr_when_direction_exists(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph()
    seen = set()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    dir_pairs = {(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'directional'}
    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', '')
        if et == 'correlational' and ((u, v) in dir_pairs or (v, u) in dir_pairs):
            continue
        k = (u, v, et)
        if k not in seen:
            H.add_edge(u, v, **dict(d))
            seen.add(k)
    return H


def collapse_multirel_edges_by_priority(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    For each directed node pair (A, B), keep only one non-hierarchy edge
    based on priority:
      directional(validated) > directional(null) > directional(hypothesized)
      > correlational(validated) > correlational(null) > correlational(hypothesized)
      > moderation(validated) > moderation(null) > moderation(hypothesized)
    Hierarchy edges are preserved separately.
    """
    priority = {
        ('directional', 'validated'): 0,
        ('directional', 'null'): 1,
        ('directional', 'hypothesized'): 2,
        ('correlational', 'validated'): 3,
        ('correlational', 'null'): 4,
        ('correlational', 'hypothesized'): 5,
        ('moderation', 'validated'): 6,
        ('moderation', 'null'): 7,
        ('moderation', 'hypothesized'): 8,
    }

    H = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)

    # Keep hierarchy edges (deduplicated by type)
    seen_h = set()
    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', '')
        if et == 'hierarchy':
            k = (u, v, et)
            if k not in seen_h:
                H.add_edge(u, v, **dict(d))
                seen_h.add(k)

    # For each non-hierarchy (u, v), choose best edge by priority
    best_by_pair = {}
    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', '')
        if et == 'hierarchy':
            continue
        validation = _validation_from_text(d.get('validation', 'hypothesized'), default='hypothesized')
        rank = priority.get((et, validation), 10_000)
        pair = (u, v)
        cur = best_by_pair.get(pair)
        cand = (rank, et, validation, dict(d))
        if cur is None or cand < cur:
            best_by_pair[pair] = cand

    for (u, v), (_rank, et, validation, attrs) in best_by_pair.items():
        attrs = dict(attrs)
        attrs['edge_type'] = et
        attrs['validation'] = validation
        H.add_edge(u, v, **attrs)

    return H


def merge_single_lv_hierarchy(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    If an HV has exactly one LV via hierarchy (HV -> LV), merge them into one node.
    Carry over non-hierarchy edges from both nodes and de-duplicate typed edges.
    """
    # Build hierarchy adjacency
    children = {}
    for u, v, d in G.edges(data=True):
        if d.get('edge_type') == 'hierarchy':
            children.setdefault(u, []).append(v)

    rename = {}
    drop_hier = set()
    for hv, lvs in children.items():
        if len(lvs) == 1:
            lv = lvs[0]
            hv_label = str(G.nodes[hv].get('label', hv)).strip()
            lv_label = str(G.nodes[lv].get('label', lv)).strip()
            merged_label = f"{hv_label} ({lv_label})"
            rename[hv] = merged_label
            rename[lv] = merged_label
            drop_hier.add((hv, lv))

    if not rename:
        # still deduplicate typed edges
        H = nx.MultiDiGraph()
        for n, d in G.nodes(data=True):
            H.add_node(n, **d)
        seen = set()
        for u, v, d in G.edges(data=True):
            et = d.get('edge_type', '')
            k = (u, v, et)
            if k not in seen:
                H.add_edge(u, v, **dict(d))
                seen.add(k)
        return H

    def mapped(n):
        return rename.get(n, n)

    H = nx.MultiDiGraph()
    seen = set()

    # Add merged/non-merged nodes
    for n, d in G.nodes(data=True):
        mn = mapped(n)
        if mn not in H:
            label = mn if n in rename else d.get('label', mn)
            H.add_node(mn, label=label)

    # Rebuild edges with merge + dedup
    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', '')
        if et == 'hierarchy' and (u, v) in drop_hier:
            continue
        mu, mv = mapped(u), mapped(v)
        if mu == mv:
            continue
        k = (mu, mv, et)
        if k not in seen:
            H.add_edge(mu, mv, **dict(d))
            seen.add(k)

    return H


def preprocess_graph_for_eval(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    # Keep logic consistent with graph_structural_eval.ipynb
    G = canonicalise_corr(G)
    G = merge_single_lv_hierarchy(G)
    G = propagate_to_upper(G)
    G = drop_nonhierarchy_between_hierarchy_linked(G)
    G = canonicalise_corr(G)
    G = drop_corr_when_direction_exists(G)
    G = collapse_multirel_edges_by_priority(G)
    return G


def to_higher_level_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Remove LV nodes (nodes that are the lower end of any hierarchy edge).
    Keep only HV nodes and all non-hierarchy edges among them.
    Hierarchy edges are removed entirely (they defined HV/LV, no longer needed).
    """
    lv_nodes = {v for u, v, d in G.edges(data=True) if d.get('edge_type') == 'hierarchy'}
    keep = set(G.nodes()) - lv_nodes
    H = nx.MultiDiGraph()
    for n in keep:
        H.add_node(n, **G.nodes[n])
    seen = set()
    for u, v, d in G.edges(data=True):
        et = d.get('edge_type', '')
        if et == 'hierarchy':
            continue
        if u in keep and v in keep:
            k = (u, v, et)
            if k not in seen:
                H.add_edge(u, v, **dict(d))
                seen.add(k)
    return H


def to_type_agnostic_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Collapse all edge types to a single 'relation' type.
    Deduplicates (u, v) pairs (keeps one edge per directed pair).
    """
    H = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    seen = set()
    for u, v, d in G.edges(data=True):
        k = (u, v)
        if k not in seen:
            H.add_edge(u, v, edge_type='relation')
            seen.add(k)
    return H


class _TimeoutError(Exception):
    pass


def _timeout_handler(_sig, _frame):
    raise _TimeoutError()


def _ordered_nodes(G_gt: nx.MultiDiGraph, strategy: str) -> list:
    nodes = list(G_gt.nodes())
    if strategy == 'degree_desc':
        return sorted(nodes, key=lambda n: G_gt.degree(n), reverse=True)
    if strategy == 'degree_asc':
        return sorted(nodes, key=lambda n: G_gt.degree(n))
    if strategy == 'label':
        return sorted(nodes, key=lambda n: str(G_gt.nodes[n].get('label', n)))
    if strategy.startswith('random_'):
        seed = int(strategy.split('_', 1)[1])
        rnd = random.Random(seed)
        out = nodes[:]
        rnd.shuffle(out)
        return out
    raise ValueError(strategy)


def mcs_f1_with_order(G_gt: nx.MultiDiGraph, G_pred: nx.MultiDiGraph, timeout: float, order_strategy: str):
    m1 = G_gt.number_of_edges()
    m2 = G_pred.number_of_edges()
    if m1 == 0 and m2 == 0:
        return {'edge_matched': 0, 'edge_p': 1.0, 'edge_r': 1.0, 'edge_f1': 1.0, 'timed_out': False, 'best_phi': {}}
    if m1 == 0 or m2 == 0:
        return {'edge_matched': 0, 'edge_p': 0.0, 'edge_r': 0.0, 'edge_f1': 0.0, 'timed_out': False, 'best_phi': {}}

    gt_nodes = _ordered_nodes(G_gt, order_strategy)
    pred_nodes = list(G_pred.nodes())
    pred_edge_set = {(u, v, d['edge_type']) for u, v, d in G_pred.edges(data=True)}
    # For correlational edges treat as undirected: also add reverse direction to lookup set
    pred_edge_set_corr_undirected = pred_edge_set | {
        (v, u, 'correlational') for u, v, et in pred_edge_set if et == 'correlational'
    }
    adj_out = {n: [(v, d['edge_type']) for _, v, d in G_gt.out_edges(n, data=True)] for n in G_gt.nodes()}
    adj_in = {n: [(u, d['edge_type']) for u, _, d in G_gt.in_edges(n, data=True)] for n in G_gt.nodes()}

    def count_em(gn, pn, phi):
        c = 0
        for v, t in adj_out.get(gn, []):
            pv = phi.get(v)
            if pv:
                lookup = pred_edge_set_corr_undirected if t == 'correlational' else pred_edge_set
                if (pn, pv, t) in lookup:
                    c += 1
        for u, t in adj_in.get(gn, []):
            pu = phi.get(u)
            if pu:
                lookup = pred_edge_set_corr_undirected if t == 'correlational' else pred_edge_set
                if (pu, pn, t) in lookup:
                    c += 1
        return c

    def greedy():
        phi = {}
        avail = list(pred_nodes)
        em = 0
        for gn in gt_nodes:
            best_c = -1
            best_p = None
            for p in avail:
                c = count_em(gn, p, phi)
                if c > best_c:
                    best_c = c
                    best_p = p
            if best_p is not None:
                phi[gn] = best_p
                avail.remove(best_p)
                em += best_c
        return em, dict(phi)

    greedy_em, greedy_phi = greedy()
    best = [greedy_em]
    best_phi = [greedy_phi]
    timed_out = False

    def branch(depth, phi, avail, cur_em, unresolved):
        if cur_em + unresolved <= best[0]:
            return
        if depth == len(gt_nodes):
            if cur_em > best[0]:
                best[0] = cur_em
                best_phi[0] = dict(phi)
            return
        gn = gt_nodes[depth]
        resolved_now = sum(1 for v, _ in adj_out.get(gn, []) if phi.get(v)) + sum(1 for u, _ in adj_in.get(gn, []) if phi.get(u))
        next_unresolved = unresolved - resolved_now
        for j, pn in enumerate(avail):
            new_em = cur_em + count_em(gn, pn, phi)
            phi[gn] = pn
            branch(depth + 1, phi, avail[:j] + avail[j + 1 :], new_em, next_unresolved)
            del phi[gn]
        branch(depth + 1, phi, avail, cur_em, next_unresolved)

    old_h = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        branch(0, {}, pred_nodes, 0, m1)
    except _TimeoutError:
        timed_out = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_h)

    em = best[0]
    ep = em / m2 if m2 > 0 else (1.0 if m1 == 0 else 0.0)
    er = em / m1 if m1 > 0 else (1.0 if m2 == 0 else 0.0)
    ef = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
    return {'edge_matched': em, 'edge_p': ep, 'edge_r': er, 'edge_f1': ef, 'timed_out': timed_out, 'best_phi': best_phi[0]}


def _per_type_f1(G_gt, G_pred, phi):
    """Given best mapping phi (gt_node -> pred_node), compute per edge-type P/R/F1."""
    EDGE_TYPES = ['hierarchy', 'directional', 'correlational', 'moderation']
    gt_edges_by_type = {}
    for u, v, d in G_gt.edges(data=True):
        gt_edges_by_type.setdefault(d.get('edge_type', ''), set()).add((u, v))
    pred_edges_by_type = {}
    for u, v, d in G_pred.edges(data=True):
        pred_edges_by_type.setdefault(d.get('edge_type', ''), set()).add((u, v))
    pred_edge_set = {(u, v, d['edge_type']) for u, v, d in G_pred.edges(data=True)}
    # Correlational treated as undirected for matching
    pred_edge_set_corr_undirected = pred_edge_set | {
        (v, u, 'correlational') for u, v, et in pred_edge_set if et == 'correlational'
    }
    result = {}
    for et in EDGE_TYPES:
        gt_et = gt_edges_by_type.get(et, set())
        pred_et = pred_edges_by_type.get(et, set())
        lookup = pred_edge_set_corr_undirected if et == 'correlational' else pred_edge_set
        matched = 0
        for gu, gv in gt_et:
            pu, pv = phi.get(gu), phi.get(gv)
            if pu and pv and (pu, pv, et) in lookup:
                matched += 1
        n_gt = len(gt_et)
        n_pred = len(pred_et)
        p = matched / n_pred if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
        r = matched / n_gt if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        result[f'{et}_matched'] = matched
        result[f'{et}_gt'] = n_gt
        result[f'{et}_pred'] = n_pred
        result[f'{et}_p'] = p
        result[f'{et}_r'] = r
        result[f'{et}_f1'] = f
    return result


def evaluate_article(args):
    aid, G_gt, G_pred, timeout_per_order, orders, best_of_orders = args
    strategies = orders if best_of_orders else [orders[0]]
    best = None
    best_strategy = None
    any_timeout = False
    for s in strategies:
        r = mcs_f1_with_order(G_gt, G_pred, timeout_per_order, s)
        any_timeout = any_timeout or r['timed_out']
        if best is None or r['edge_f1'] > best['edge_f1'] or (r['edge_f1'] == best['edge_f1'] and r['edge_matched'] > best['edge_matched']):
            best = r
            best_strategy = s
    phi = best.pop('best_phi', {})
    best.update(_per_type_f1(G_gt, G_pred, phi))
    best['article_id'] = aid
    best['gt_n_edges'] = G_gt.number_of_edges()
    best['pred_n_edges'] = G_pred.number_of_edges()
    best['strategy'] = best_strategy
    best['any_timeout'] = any_timeout
    return best


# ── Validation-aware variants ─────────────────────────────────────────────────
# Edge key: (u, v, edge_type, validation).  Correlational still undirected.

def mcs_f1_with_validation(G_gt: nx.MultiDiGraph, G_pred: nx.MultiDiGraph,
                            timeout: float, order_strategy: str):
    """Like mcs_f1_with_order but an edge matches only when both type AND
    validation status agree (validated / null / hypothesized)."""
    m1 = G_gt.number_of_edges()
    m2 = G_pred.number_of_edges()
    if m1 == 0 and m2 == 0:
        return {'edge_matched': 0, 'edge_p': 1.0, 'edge_r': 1.0, 'edge_f1': 1.0,
                'timed_out': False, 'best_phi': {}}
    if m1 == 0 or m2 == 0:
        return {'edge_matched': 0, 'edge_p': 0.0, 'edge_r': 0.0, 'edge_f1': 0.0,
                'timed_out': False, 'best_phi': {}}

    gt_nodes = _ordered_nodes(G_gt, order_strategy)
    pred_nodes = list(G_pred.nodes())

    pred_edge_set = {(u, v, d['edge_type'], d.get('validation', 'hypothesized'))
                     for u, v, d in G_pred.edges(data=True)}
    pred_edge_set_corr_undirected = pred_edge_set | {
        (v, u, 'correlational', val)
        for u, v, et, val in pred_edge_set if et == 'correlational'
    }

    adj_out = {n: [(v, d['edge_type'], d.get('validation', 'hypothesized'))
                   for _, v, d in G_gt.out_edges(n, data=True)]
               for n in G_gt.nodes()}
    adj_in  = {n: [(u, d['edge_type'], d.get('validation', 'hypothesized'))
                   for u, _, d in G_gt.in_edges(n, data=True)]
               for n in G_gt.nodes()}

    def count_em(gn, pn, phi):
        c = 0
        for v, t, val in adj_out.get(gn, []):
            pv = phi.get(v)
            if pv:
                lookup = pred_edge_set_corr_undirected if t == 'correlational' else pred_edge_set
                if (pn, pv, t, val) in lookup:
                    c += 1
        for u, t, val in adj_in.get(gn, []):
            pu = phi.get(u)
            if pu:
                lookup = pred_edge_set_corr_undirected if t == 'correlational' else pred_edge_set
                if (pu, pn, t, val) in lookup:
                    c += 1
        return c

    def greedy():
        phi = {}
        avail = list(pred_nodes)
        em = 0
        for gn in gt_nodes:
            best_c, best_p = -1, None
            for p in avail:
                c = count_em(gn, p, phi)
                if c > best_c:
                    best_c, best_p = c, p
            if best_p is not None:
                phi[gn] = best_p
                avail.remove(best_p)
                em += best_c
        return em, dict(phi)

    greedy_em, greedy_phi = greedy()
    best = [greedy_em]
    best_phi = [greedy_phi]
    timed_out = False

    def branch(depth, phi, avail, cur_em, unresolved):
        if cur_em + unresolved <= best[0]:
            return
        if depth == len(gt_nodes):
            if cur_em > best[0]:
                best[0] = cur_em
                best_phi[0] = dict(phi)
            return
        gn = gt_nodes[depth]
        resolved_now = (sum(1 for v, _, __ in adj_out.get(gn, []) if phi.get(v)) +
                        sum(1 for u, _, __ in adj_in.get(gn, [])  if phi.get(u)))
        next_unresolved = unresolved - resolved_now
        for j, pn in enumerate(avail):
            new_em = cur_em + count_em(gn, pn, phi)
            phi[gn] = pn
            branch(depth + 1, phi, avail[:j] + avail[j + 1:], new_em, next_unresolved)
            del phi[gn]
        branch(depth + 1, phi, avail, cur_em, next_unresolved)

    old_h = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        branch(0, {}, pred_nodes, 0, m1)
    except _TimeoutError:
        timed_out = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_h)

    em = best[0]
    ep = em / m2 if m2 > 0 else (1.0 if m1 == 0 else 0.0)
    er = em / m1 if m1 > 0 else (1.0 if m2 == 0 else 0.0)
    ef = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
    return {'edge_matched': em, 'edge_p': ep, 'edge_r': er, 'edge_f1': ef,
            'timed_out': timed_out, 'best_phi': best_phi[0]}


def _per_type_validation_f1(G_gt, G_pred, phi):
    """Per-type AND per-validation-status F1 given node mapping phi."""
    EDGE_TYPES   = ['hierarchy', 'directional', 'correlational', 'moderation']
    VALIDATIONS  = ['validated', 'null', 'hypothesized']

    pred_edge_set = {(u, v, d['edge_type'], d.get('validation', 'hypothesized'))
                     for u, v, d in G_pred.edges(data=True)}
    pred_edge_set_corr = pred_edge_set | {
        (v, u, 'correlational', val)
        for u, v, et, val in pred_edge_set if et == 'correlational'
    }

    # index GT/pred by (type, validation)
    gt_by_tv   = {}
    pred_by_tv = {}
    for u, v, d in G_gt.edges(data=True):
        key = (d.get('edge_type', ''), d.get('validation', 'hypothesized'))
        gt_by_tv.setdefault(key, set()).add((u, v))
    for u, v, d in G_pred.edges(data=True):
        key = (d.get('edge_type', ''), d.get('validation', 'hypothesized'))
        pred_by_tv.setdefault(key, set()).add((u, v))

    result = {}
    for et in EDGE_TYPES:
        for val in VALIDATIONS:
            tv   = (et, val)
            gt_s = gt_by_tv.get(tv, set())
            pd_s = pred_by_tv.get(tv, set())
            lookup = pred_edge_set_corr if et == 'correlational' else pred_edge_set
            matched = sum(
                1 for gu, gv in gt_s
                if phi.get(gu) and phi.get(gv) and (phi[gu], phi[gv], et, val) in lookup
            )
            n_gt, n_pred = len(gt_s), len(pd_s)
            p = matched / n_pred if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
            r = matched / n_gt  if n_gt   > 0 else (1.0 if n_pred == 0 else 0.0)
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            prefix = f'{et}_{val}'
            result[f'{prefix}_matched'] = matched
            result[f'{prefix}_gt']      = n_gt
            result[f'{prefix}_pred']    = n_pred
            result[f'{prefix}_p']       = p
            result[f'{prefix}_r']       = r
            result[f'{prefix}_f1']      = f
    return result


def evaluate_article_with_validation(args):
    """Drop-in replacement for evaluate_article that requires type+validation match."""
    aid, G_gt, G_pred, timeout_per_order, orders, best_of_orders = args
    strategies = orders if best_of_orders else [orders[0]]
    best = None
    best_strategy = None
    any_timeout = False
    for s in strategies:
        r = mcs_f1_with_validation(G_gt, G_pred, timeout_per_order, s)
        any_timeout = any_timeout or r['timed_out']
        if best is None or r['edge_f1'] > best['edge_f1'] or \
                (r['edge_f1'] == best['edge_f1'] and r['edge_matched'] > best['edge_matched']):
            best = r
            best_strategy = s
    phi = best.pop('best_phi', {})
    best.update(_per_type_validation_f1(G_gt, G_pred, phi))
    best['article_id']   = aid
    best['gt_n_edges']   = G_gt.number_of_edges()
    best['pred_n_edges'] = G_pred.number_of_edges()
    best['strategy']     = best_strategy
    best['any_timeout']  = any_timeout
    return best
