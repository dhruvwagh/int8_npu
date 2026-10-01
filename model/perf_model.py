#!/usr/bin/env python3
"""
File: perf_model.py
Description: Cycle-level performance model of the pipelined NPU (milestone 13
             microarchitecture). It models control and data movement only:
             fetch engines, the shared memory port and its arbiter, the
             sequencer's fetch/compute controllers, the capture wave, drain,
             epilogue, obuf queue and writeback. No arithmetic is done.

             Every state element is updated once per cycle from the previous
             cycle's values, the same register-transfer timing the RTL uses,
             so the model can be checked cycle-for-cycle against the RTL
             performance counters at the RTL's configuration (16x16 array,
             16-byte port). It then answers questions the RTL cannot without
             redesign: other port widths, array sizes and K tile depths.

Generalisations beyond the RTL (used only for design-space exploration):
  - port_bytes != arr: a panel word of `arr` bytes takes arr/port_bytes beats
    (narrow port) or several panel words share a beat (wide port).
  - arr != 16: panel words are `arr` bytes; bias is 4*arr bytes per panel.
Author: Dhruv Wagh
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ArchCfg:
    arr: int = 16              # ARR_M = ARR_N
    k_tile: int = 128          # K_TILE_MAX
    port_bytes: int = 16       # tensor memory port width (bytes/beat)
    max_outstanding: int = 8   # arbiter tag FIFO depth
    mem_lat: int = 1           # memory read latency (cycles after accept)
    mem_jit: int = 0           # extra random latency 0..mem_jit
    mem_ready_pct: int = 100   # probability the memory accepts a request
    seed: int = 1

    def beats(self, panel_words: int) -> int:
        """Port beats to move `panel_words` words of `arr` bytes each."""
        return max(1, math.ceil(panel_words * self.arr / self.port_bytes))


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
class Fetch:
    """npu_fetch: issues `total` read beats, done pulse after the last response."""

    def __init__(self):
        self.run = False
        self.req_cnt = 0
        self.resp_cnt = 0
        self.total = 0
        self.done = False           # registered pulse

    def req(self) -> bool:
        return self.run and self.req_cnt < self.total

    def next(self, start: bool, total: int, granted: bool, rsp: bool):
        n = dict(run=self.run, req_cnt=self.req_cnt, resp_cnt=self.resp_cnt,
                 total=self.total, done=False)
        if not self.run:
            if start and total > 0:
                n.update(run=True, req_cnt=0, resp_cnt=0, total=total)
        else:
            if granted:
                n["req_cnt"] = self.req_cnt + 1
            if rsp:
                n["resp_cnt"] = self.resp_cnt + 1
                if self.resp_cnt + 1 == self.total:
                    n.update(run=False, done=True)
        return n


class Writeback:
    """npu_wb: obuf reads run ahead of memory writes through a 2-entry FIFO."""

    def __init__(self):
        self.active = False
        self.total = 0
        self.rd_cnt = 0
        self.wr_cnt = 0
        self.rd_inflight = False
        self.fifo = 0

    def mem_req(self) -> bool:
        return self.active and self.fifo > 0

    def next(self, start: bool, total: int, granted: bool):
        pop = self.mem_req() and granted
        occ_next = self.fifo + int(self.rd_inflight) - int(pop)
        rd_en = self.active and self.rd_cnt < self.total and occ_next < 2
        n = dict(active=self.active, total=self.total,
                 rd_cnt=self.rd_cnt + int(rd_en), wr_cnt=self.wr_cnt + int(pop),
                 rd_inflight=rd_en, fifo=self.fifo + int(self.rd_inflight) - int(pop))
        if not self.active:
            if start and total > 0:
                n.update(active=True, total=total, rd_cnt=0, wr_cnt=0)
        elif pop and self.wr_cnt + 1 == self.total:
            n["active"] = False
        return n


@dataclass
class Tile:
    m_len: int
    n_len: int
    k_len: int
    last_k: bool
    new_wgt: bool
    new_bias: bool
    last_tile: bool
    slot: int
    rel_slot: bool


def tile_sequence(M: int, N: int, K: int, cfg: ArchCfg) -> List[Tile]:
    """Loop nest of npu_sequencer (fetch side), including residency decisions."""
    a, kt = cfg.arr, cfg.k_tile
    k_split = K > kt and N > a
    tiles, key, slot = [], None, 0
    m0 = n0 = k0 = 0
    while True:
        last_m, last_n, last_k = m0 + a >= M, n0 + a >= N, k0 + kt >= K
        new_wgt = key != (m0, k0)
        key = (m0, k0)
        tiles.append(Tile(min(a, M - m0), min(a, N - n0), min(kt, K - k0), last_k, new_wgt,
                          k0 == 0 and n0 == 0, last_m and last_n and last_k, slot,
                          last_n and last_k))
        if last_m and last_n and last_k:
            return tiles
        if k_split:
            if not last_k:
                k0 += kt
            else:
                k0 = 0
                if not last_n:
                    n0 += a
                else:
                    n0, m0, slot = 0, m0 + a, slot ^ 1
        else:
            if not last_n:
                n0 += a
            else:
                n0 = 0
                if not last_k:
                    k0 += kt
                else:
                    k0, m0, slot = 0, m0 + a, slot ^ 1


# ----------------------------------------------------------------------------
# Whole-NPU model
# ----------------------------------------------------------------------------
def simulate_layer(M: int, N: int, K: int, cfg: ArchCfg = None, max_cycles: int = 10_000_000) -> Dict:
    cfg = cfg or ArchCfg()
    rng = random.Random(cfg.seed)
    tiles = tile_sequence(M, N, K, cfg)
    A = cfg.arr
    CAP_DELAY = A + 2                  # ARR_M + OPERAND_LAT
    bias_beats = cfg.beats(4)          # 4*arr bytes = 4 panel words

    # ---- state (register values visible in the current cycle) ----
    fw, fa, fb = Fetch(), Fetch(), Fetch()
    wb = Writeback()
    tags: List[int] = []               # outstanding read requester ids (0=wgt,1=act,2=bias)
    rsp_q: List[list] = []             # [ready_cycle, requester]
    last_rsp = -1
    mem_ready = cfg.mem_ready_pct >= 100
    mem_rsp = None                     # requester id whose response is visible this cycle

    S = dict(
        fstate="ISSUE", cstate="CLEAR", ti=0,
        mbox=None, f_tile=None, f_pend=set(),
        wgt_full=False, bias_busy=[False, False],
        start_w=0, start_a=0, start_b=0,          # registered fetch_*_start (beats)
        c_tile=None, c_cnt=0,
        cap_pending=False, cap_timer=0, cap_tile=None,
        drain_start=False, drn_active=False, drn_tile=None,
        drain_col=-1,                              # drain column being output, -1 idle
        drain_done=False,
        epi_pipe=[None, None, None, None],         # column index in epilogue stages 0, 1, 2a, 2b
        drain_inflight=False, dr_tile=None,
        wbq=None, wb_start=None,
        done=False,
    )
    stats = dict(cycles=0, macs=0, stall=0, stream=0, waves=0, port_read_beats=0,
                 port_write_beats=0, port_idle=0)

    cyc = 0
    while not S["done"]:
        cyc += 1
        if cyc > max_cycles:
            raise RuntimeError("perf model did not finish")
        stats["cycles"] += 1

        # ------------------------------------------------------------------
        # Combinational: port arbitration (priority wb > wgt > act > bias)
        # ------------------------------------------------------------------
        reqs = [("wb", wb.mem_req(), True), ("w", fw.req(), False),
                ("a", fa.req(), False), ("b", fb.req(), False)]
        grant = None
        for name, r, is_wr in reqs:
            if r and (is_wr or len(tags) < cfg.max_outstanding):
                grant = (name, is_wr)
                break
        accepted = grant is not None and mem_ready
        g = grant[0] if accepted else None
        if accepted:
            if grant[1]:
                stats["port_write_beats"] += 1
            else:
                stats["port_read_beats"] += 1
        else:
            stats["port_idle"] += 1

        # Response routing (visible this cycle)
        rsp = {"w": False, "a": False, "b": False}
        if mem_rsp is not None:
            rsp[mem_rsp] = True
            tags.pop(0)

        # ------------------------------------------------------------------
        # Output path (combinational view of current registers)
        # ------------------------------------------------------------------
        epi_out_col = S["epi_pipe"][-1]
        obuf_swap = epi_out_col == A - 1
        drain_ready = not S["drain_inflight"] and S["wbq"] is None
        out_idle = drain_ready and S["wb_start"] is None and not wb.active

        # ------------------------------------------------------------------
        # Next state
        # ------------------------------------------------------------------
        N_ = dict(S)
        N_["start_w"] = N_["start_a"] = N_["start_b"] = 0
        N_["drain_start"] = False
        N_["wb_start"] = None
        f_pend = set(S["f_pend"])
        for k, f in (("w", fw), ("a", fa), ("b", fb)):
            if f.done:
                f_pend.discard(k)

        # ---- FETCH side ----
        if S["fstate"] == "ISSUE":
            t = tiles[S["ti"]]
            can = (S["mbox"] is None and not (t.new_wgt and S["wgt_full"])
                   and not (t.new_bias and S["bias_busy"][t.slot]))
            if can:
                N_["start_a"] = cfg.beats(t.k_len)
                f_pend.add("a")
                if t.new_wgt:
                    N_["start_w"] = cfg.beats(t.k_len)
                    f_pend.add("w")
                    N_["wgt_full"] = True
                if t.new_bias:
                    N_["start_b"] = bias_beats
                    f_pend.add("b")
                    bb = list(N_["bias_busy"]); bb[t.slot] = True; N_["bias_busy"] = bb
                N_["f_tile"] = t
                N_["ti"] = S["ti"] + 1
                N_["fstate"] = "WAIT"
        elif S["fstate"] == "WAIT":
            if not S["f_pend"]:
                N_["mbox"] = S["f_tile"]
                N_["fstate"] = "DONE" if S["f_tile"].last_tile else "ISSUE"
        elif S["fstate"] == "DONE":
            if S["cstate"] == "IDLE":
                N_["fstate"] = "IDLE"
        N_["f_pend"] = f_pend

        # ---- COMPUTE side ----
        c, ct = S["cstate"], S["c_tile"]
        wave_ok = not S["cap_pending"]
        last_read = c == "STREAM" and S["c_cnt"] + 1 == ct.k_len
        pop = S["mbox"] is not None and (
            c == "WAIT" or (last_read and not ct.last_k) or
            (c == "WAVE" and wave_ok and not ct.last_tile))
        if c == "CLEAR":
            N_["cstate"] = "WAIT"
        elif c == "WAIT":
            if S["mbox"] is None:
                stats["stall"] += 1
        elif c == "STREAM":
            stats["macs"] += ct.m_len * ct.n_len
            stats["stream"] += 1
            N_["c_cnt"] = S["c_cnt"] + 1
            if last_read:
                if ct.last_k:
                    N_["cstate"] = "WAVE"
                elif S["mbox"] is None:
                    N_["cstate"] = "WAIT"
        elif c == "WAVE":
            if wave_ok:
                stats["waves"] += 1
                N_["cap_pending"] = True
                N_["cap_timer"] = CAP_DELAY
                N_["cap_tile"] = ct
                if ct.last_tile:
                    N_["cstate"] = "FINISH"
                elif S["mbox"] is None:
                    N_["cstate"] = "WAIT"
            else:
                stats["stall"] += 1
        elif c == "FINISH":
            if (not S["cap_pending"] and not S["drn_active"] and not S["drain_start"]
                    and out_idle):
                N_["cstate"] = "IDLE"
        if pop:
            N_["mbox"] = None
            if S["mbox"].new_wgt:
                N_["wgt_full"] = False
            N_["c_tile"] = S["mbox"]
            N_["c_cnt"] = 0
            N_["cstate"] = "STREAM"

        # ---- drain scheduling ----
        if S["cap_pending"] and S["cap_timer"] > 1:
            N_["cap_timer"] = S["cap_timer"] - 1
        if S["cap_pending"] and S["cap_timer"] == 1 and drain_ready:
            N_["drain_start"] = True
            N_["cap_pending"] = False
            N_["drn_active"] = True
            N_["drn_tile"] = S["cap_tile"]
        if S["drain_done"] and S["drn_active"]:
            N_["drn_active"] = False
            if S["drn_tile"].rel_slot:
                bb = list(N_["bias_busy"]); bb[S["drn_tile"].slot] = False; N_["bias_busy"] = bb

        # ---- drain module / epilogue pipeline ----
        N_["drain_done"] = False
        if S["drain_start"] and S["drain_col"] < 0:
            N_["drain_col"] = 0
        elif S["drain_col"] >= 0:
            if S["drain_col"] == A - 1:
                N_["drain_col"] = -1
                N_["drain_done"] = True
            else:
                N_["drain_col"] = S["drain_col"] + 1
        N_["epi_pipe"] = [S["drain_col"] if S["drain_col"] >= 0 else None] + S["epi_pipe"][:-1]

        # ---- output queue / writeback start ----
        if S["drain_start"]:
            N_["drain_inflight"] = True
            N_["dr_tile"] = S["drn_tile"]
        if obuf_swap:
            N_["drain_inflight"] = False
            N_["wbq"] = S["dr_tile"]
        if S["wbq"] is not None and not wb.active and S["wb_start"] is None:
            N_["wbq"] = None if not obuf_swap else N_["wbq"]
            N_["wb_start"] = S["wbq"]

        if S["fstate"] != "IDLE" or S["cstate"] != "IDLE":
            pass
        else:
            stats["cycles"] -= 1
        if N_["fstate"] == "IDLE" and N_["cstate"] == "IDLE":
            N_["done"] = True

        # ---- blocks with their own registers ----
        fw_n = fw.next(S["start_w"] > 0, S["start_w"], g == "w", rsp["w"])
        fa_n = fa.next(S["start_a"] > 0, S["start_a"], g == "a", rsp["a"])
        fb_n = fb.next(S["start_b"] > 0, S["start_b"], g == "b", rsp["b"])
        wst = S["wb_start"]
        wb_n = wb.next(wst is not None, cfg.beats(wst.n_len) if wst is not None else 0, g == "wb")

        # ---- memory ----
        if accepted and not grant[1]:
            t_ready = cyc + cfg.mem_lat + (rng.randint(0, cfg.mem_jit) if cfg.mem_jit else 0)
            t_ready = max(t_ready, last_rsp + 1)
            last_rsp = t_ready
            rsp_q.append([t_ready, g])
            tags.append(g)
        mem_rsp = None
        if rsp_q and rsp_q[0][0] <= cyc:
            mem_rsp = rsp_q.pop(0)[1]
        mem_ready = (rng.randrange(100) < cfg.mem_ready_pct) and len(rsp_q) < 62

        # ---- commit ----
        S = N_
        for obj, n in ((fw, fw_n), (fa, fa_n), (fb, fb_n), (wb, wb_n)):
            obj.__dict__.update(n)

    stats["util"] = stats["macs"] / (stats["cycles"] * A * A)
    stats["tiles"] = len(tiles)
    return stats


# ----------------------------------------------------------------------------
# Analytic bounds (for explaining where cycles go)
# ----------------------------------------------------------------------------
def layer_bounds(M: int, N: int, K: int, cfg: ArchCfg = None) -> Dict:
    """Lower bounds on cycles from each resource, ignoring overlaps' imperfections."""
    cfg = cfg or ArchCfg()
    tiles = tile_sequence(M, N, K, cfg)
    compute = sum(t.k_len for t in tiles) + sum(t.last_k for t in tiles)   # stream + waves
    port = sum(cfg.beats(t.k_len) * (1 + t.new_wgt) + (cfg.beats(4) if t.new_bias else 0)
               + (cfg.beats(t.n_len) if t.last_k else 0) for t in tiles)
    drain = sum(t.last_k for t in tiles) * (cfg.arr + 2)
    ideal = math.ceil(M * N * K / (cfg.arr * cfg.arr))
    return dict(compute=compute, port=port, drain=drain, ideal_macs=ideal,
                bound=max(compute, port, drain))


MNIST_LAYERS = [("conv1", 8, 784, 9), ("conv2", 16, 196, 72), ("fc", 10, 1, 784)]


if __name__ == "__main__":
    cfg = ArchCfg()
    tot = 0
    for name, M, N, K in MNIST_LAYERS:
        r = simulate_layer(M, N, K, cfg)
        b = layer_bounds(M, N, K, cfg)
        tot += r["cycles"]
        print(f"{name:6s} cycles={r['cycles']:6d} util={100 * r['util']:5.1f}%  "
              f"bounds: compute={b['compute']} port={b['port']} drain={b['drain']}")
    print(f"total  {tot} cycles/image")
