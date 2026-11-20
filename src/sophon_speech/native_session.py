# Original Session statements: Adventists AI Apache-2.0, pinned by upstream.json.
# Eager original-E2 profile; device-only transfers are explicit new Source.
import os,re,time,math,threading
from contextlib import nullcontext as _NULL
import numpy as np
import torch
# L, MK, dev, decoder_device and PL are injected by this private runtime factory.
T32_EV=True;EV_GAIN=1.;K0=2;RHO=1.;FUSED=False;M2=False;SCHED=False;E2B=False
READ1=False;MINNF=0;TRACE=False;FIRST=4;NEXT=6;HOLD=2;PRE=3600
WIN=60;XF=8;NT_FIRST=10;NT=10;PAR_E2=False;FIRST_EXCL=True;PAR_AHEAD=1.
DBG=False;RAMP=[2,3,4];TAIL_TOK=3;DIT_STREAM=True;CAP_C=1280;CAP_T=4096
JUNK = re.compile(r"[*#`>_~\[\]{}<>|\\/]|[\U0001F000-\U0001FFFF☀-➿]")
QUOTE = re.compile(r"[“”「」『』\"《》()（）【】]")
def _pend0(s):
    p=getattr(s,'v2_pend',None)
    return p is None or p.shape[2]==0
class Session:
    def __init__(s, core, lang, voice_key, emit):
        s.c = core; s.lang = lang; s.emit = emit; s.voice = core.voices[voice_key] if voice_key in core.voices else core.voices.get(f"brand_{lang}", core.voices["brand_zh"])
        s.text = ""; s.final = False; s.cancelled = False; s.finished = False; s.ev = threading.Event()
        s._evv = None; s._evl = None; s._evch = []; s._evlead = 0; s._ev0 = False
        s.ids = []; s.Kc, s.Vc = (list(x) for x in core.Kp_l.get(lang, (core.Kp, core.Vp))); s.Lc = PL
        s.cache = [None] * L.NL; s.x = core.x0.clone(); s.r = 0; s.nf = 0; s.p = 0; s.run_ = 0; s.done = False
        s.win_start = 0; s.tail = None
        s.frames = []; s.mel_out = 0; s.mel_cum = None; s.speech_off = 0; s.started = False; s.pending = np.zeros(0, np.float32)
        s.g = None; s.t_start = time.time()
        if core.gs is not None:
            for _ in range(400):
                with core.lock:
                    free = [g for g in core.pool if g.owner is None or g.owner.finished]
                    if free: s.g = free[0]; s.g.reset(s); break
                    live = [g for g in core.pool if not g.owner.cancelled]
                    if live: min(live, key=lambda g: g.owner.t_start).owner.cancel()
                time.sleep(0.005)
            if s.g is None: raise RuntimeError("no free talker slot")
        s.stats = dict(t0=time.perf_counter(), t_first_audio=None, n_tok=0, n_frames=0, n_step=0, ms_ctx=0.0, ms_talker=0.0, ms_e2=0.0)
    def add_text(s, t):
        if "t_text0" not in s.stats: s.stats["t_text0"] = round(time.perf_counter() - s.stats["t0"], 3)
        s.text += t; s.ev.set()
    def finish(s): s.final = True; s.ev.set()
    def cancel(s): s.cancelled = True; s.ev.set()
    def _ev_clean(s):
        raw = s.text
        if not s.final: raw = re.sub(r"(\[[a-z_]*|</?[a-z_]*)$", "", raw)
        ch, lead = MK.walk(raw); keep = []
        for c_ in ch:
            if JUNK.fullmatch(c_["c"]) or QUOTE.fullmatch(c_["c"]):
                if c_["a"] and keep and not keep[-1]["a"]: keep[-1]["a"] = c_["a"]
                continue
            keep.append(c_)
        while keep and keep[0]["c"].isspace(): keep.pop(0)
        if s.final:
            while keep and keep[-1]["c"].isspace(): keep.pop()
        s._evch = keep; s._evlead = lead
        return "".join(c_["c"] for c_ in keep)
    def _ev_build(s, clean):
        enc = s.c.tok(clean, add_special_tokens=False, return_offsets_mapping=True); n = len(enc.input_ids); ch = s._evch
        v = torch.zeros(n + 1, L.EVD_); v[:, L.EV_NC + 2] = 1.0
        for k, (b0, b1) in enumerate(enc.offset_mapping):
            for c in range(b0, b1):
                if ch[c]["a"] and ch[c]["a"] <= L.EV_NC: v[k, ch[c]["a"] - 1] = 1.0
                v[k, L.EV_NC] = max(v[k, L.EV_NC], ch[c]["sp"][0]); v[k, L.EV_NC + 1] = max(v[k, L.EV_NC + 1], ch[c]["sp"][1])
        s._evv = L.ev_shift(v, list(enc.input_ids) + [s.c.IM])
        lv = None
        if s._evlead and s._evlead <= L.EV_NC: lv = torch.zeros(L.EVD_); lv[s._evlead - 1] = 1.0; lv[L.EV_NC + 2] = 1.0
        s._evl = lv
    @torch.no_grad()
    def _ev_add(s):
        m = s.c.m
        if s.r >= 1:
            if s._evv is not None and s.r - 1 < len(s._evv):
                e_ = m.ev_embed(s._evv[s.r - 1].to(dev).float())
                if EV_GAIN != 1.0 and not L.EV_LGAIN:
                    v0_ = s._evv[s.r - 1].to(dev).float().clone(); v0_[:L.EV_NC] = 0; fb_ = m.ev_up(v0_); e_ = fb_ + EV_GAIN * (e_ - fb_)
                s.x = s.x + e_[None, None]
        elif s._evl is not None: s.x = s.x + m.ev_embed(s._evl.to(dev).float())[None, None]
    @torch.no_grad()
    def _commit(s):
        if T32_EV: clean = s._ev_clean()
        else: clean = QUOTE.sub("", JUNK.sub("", s.text)); clean = clean.strip() if s.final else clean.lstrip()
        if not s.final and len(s.ids) >= s.r + K0 + int(math.floor(RHO * s.nf)) + 1 + int(os.environ.get("T32_AHEAD", "6")): return False
        ids_all = s.c.tok(clean, add_special_tokens=False).input_ids
        target = ids_all + [s.c.IM] if s.final else ids_all[: max(0, len(ids_all) - TAIL_TOK)]
        if T32_EV: s._ev_build(clean)
        if len(target) <= len(s.ids): return False
        if target[: len(s.ids)] != s.ids: target = s.ids + target[len(s.ids):]
        new = target[len(s.ids):]; t0 = time.perf_counter()
        with (_NULL() if (M2 and s.g is not None) else s.c.lock):
            if s.g is not None:
                if s.g.owner is not s or s.cancelled: s.cancelled = True; return False
                if s.Lc + len(new) > CAP_C: new = new[: CAP_C - s.Lc]; target = s.ids + new
                if not new: return False
                s.g.ctx_extend(new, s.Lc)
            else: s.Kc, s.Vc = s.c.ctx_extend(s.Kc, s.Vc, new, s.Lc)
        s.Lc += len(new); s.ids = list(target); s.stats["n_tok"] = len(s.ids); s.stats["ms_ctx"] += 1000 * (time.perf_counter() - t0)
        return True
    @torch.no_grad()
    def _steps(s, budget=6):
        c = s.c; m = c.m; n = len(s.ids); fin = s.final and n and s.ids[-1] == c.IM; did = 0; t0 = time.perf_counter()
        mf = int(n * 4 + 20)
        while did < budget and not s.done and not s.cancelled:
            if FIRST_EXCL and s.mel_out == 0 and len(s.frames) >= FIRST + HOLD: break
            need = s.r + K0 + int(math.floor(RHO * s.nf))
            if not fin and n < need + 1: break
            if T32_EV and s.p == 0 and not s._ev0: s._ev0 = True; s._ev_add()
            vis = min(n, need); P = PL
            cm = torch.zeros(1, 1, 1, s.Lc + s.p + 1, dtype=torch.bool, device=dev); cm[..., : P + vis] = True; cm[..., s.Lc:] = True
            if s.g is not None:
                if s.g.owner is not s or s.p >= CAP_T - 1: s.cancelled = True; break
                with (_NULL() if M2 else c.lock):
                    if s.g.owner is not s or s.cancelled: s.cancelled = True; break
                    y = s.g.step(s.x, P, vis, s.p)[:, -1].float()
            else:
                with c.lock, torch.autocast("cuda", dtype=torch.bfloat16):
                    y, new = m.trunk(s.x.float(), torch.tensor([[P + s.p]], device=dev), s.Kc, s.Vc, cm, cache=s.cache)
                s.cache = [(nk, nv) if cc is None else (torch.cat([cc[0], nk], 2), torch.cat([cc[1], nv], 2)) for cc, (nk, nv) in zip(s.cache, new)]
                y = y[:, -1].float()
            lg = m.dec(y)[0].float(); s.p += 1; did += 1
            if s.r >= n: lg[L.KA] = -1e9
            if s.r < n or s.nf == 0 or not fin: lg[L.KS] = -1e9
            if s.nf < MINNF: lg[L.KS] = -1e9
            if READ1 and s.r == 0: lg[L.KF] = -1e9
            cap = s.nf >= mf or (L.STALL > 0 and s.run_ >= L.STALL and s.r < n)
            if cap: lg[L.KF] = -1e9
            if cap and s.r >= n and fin: lg[L.KS] = 1e9
            a = int(lg.argmax())
            if TRACE: s.__dict__.setdefault('trace', []).append('SAF'[[L.KS, L.KA, L.KF].index(a)] if a in (L.KS, L.KA, L.KF) else '?')
            if a == L.KS: s.done = True; break
            if a == L.KF:
                f = m.fm.from_z(m.reg(y).float()); s.frames.append(f[0]); s.x = f[:, None]; s.nf += 1; s.run_ += 1
                if s.nf in (1, 4) and f"t_f{s.nf}" not in s.stats: s.stats[f"t_f{s.nf}"] = round(time.perf_counter() - s.stats["t0"], 3); s.stats[f"tok_at_f{s.nf}"] = len(s.ids)
            else:
                s.x = (c.TH.E[s.ids[s.r]].float() + c.zb)[None, None]
                if L.HID and s.g is not None and s.g.H is not None: s.x = s.x + m.hid_up(s.g.H[PL + s.r].float())[None, None]
                s.r += 1; s.run_ = 0
            if T32_EV: s._ev_add()
            if s.p > mf + n + 5 and fin: s.done = True
        s.stats["ms_talker"] += 1000 * (time.perf_counter() - t0); s.stats["n_step"] = s.p; s.stats["n_frames"] = s.nf
        return did > 0
    @torch.no_grad()
    def _render(s):
        if os.environ.get("E2V2", "0") == "1": return s._render_v2()
        c = s.c; K = len(s.frames); last = s.done; ws = s.win_start; we = min(K, ws + WIN)
        full = (we == ws + WIN) and not (last and we == K)
        fin = we if (last and we == K) else (we if full else we - HOLD)
        final = last and we == K
        emit_to = we if final else min(fin, ws + WIN - XF)
        nr = s.stats.get("n_render", 0); need = FIRST if s.mel_out == 0 else (RAMP[nr - 1] if 0 < nr <= len(RAMP) else NEXT)
        if not full and not final and (emit_to * 8 - s.mel_out < need * 8): return False
        t0 = time.perf_counter()
        if "t_r0" not in s.stats: s.stats["t_r0"] = round(t0 - s.stats["t0"], 3)
        import contextlib
        with (_NULL() if E2B else c.e2lock), (torch.cuda.stream(c.e2stream) if s.par_on else contextlib.nullcontext()):
            if s.par_on: c.e2stream.wait_stream(torch.cuda.default_stream())
            fr = torch.stack(s.frames[ws:we])[None].to(decoder_device).float()
            _tA = time.perf_counter(); _, mux, _ = c.m2(fr); mux = mux.float()[:, :, : (fin - ws) * 8]
            if DBG and s.mel_out == 0: torch.cuda.synchronize(); s.stats["r0_m2"] = round(1000 * (time.perf_counter() - _tA))
            torch.manual_seed(0); cond = torch.zeros_like(mux); off = 0
            PR = c.prompts.get(s.lang) or c.prompts.get("zh")
            if PR is not None: mur, mr = PR; off = mur.shape[2]; mux = torch.cat([mur, mux], 2); cond = torch.zeros_like(mux); cond[:, :, :off] = mr
            mel, _ = c.dec(mu=mux, mask=torch.ones(1, 1, mux.shape[2], device=decoder_device), n_timesteps=(NT_FIRST if not s.par_on else NT), spks=c.aff(s.voice), cond=cond, streaming=DIT_STREAM)
            mel = mel[:, :, off:]
            if DBG and s.mel_out == 0: torch.cuda.synchronize(); s.stats["r0_dec"] = round(1000 * (time.perf_counter() - _tA))
            if s.tail is not None:
                o = min(s.tail.shape[2], mel.shape[2]); r_ = torch.linspace(0, 1, o, device=decoder_device)
                mel = torch.cat([s.tail[:, :, :o] * (1 - r_) + mel[:, :, :o] * r_, mel[:, :, o:]], 2)
            a, b = s.mel_out - ws * 8, emit_to * 8 - ws * 8
            new = mel[:, :, a:b] if b > a else mel[:, :, :0]
            if new.shape[2]: s.mel_out = emit_to * 8; s.mel_cum = new if s.mel_cum is None else torch.cat([s.mel_cum, new], 2)
            if full:
                s.tail = mel[:, :, (WIN - XF) * 8: WIN * 8].clone(); s.win_start = ws + WIN - XF
            done_all = final
            w, _ = c.hift.inference(speech_feat=s.mel_cum, finalize=done_all) if s.mel_cum is not None else (torch.zeros(1, 0, device=decoder_device), None)
        w = w.float()[0].cpu().numpy(); w_new = w[s.speech_off:]; s.speech_off = len(w)
        if DBG and "r0_all" not in s.stats: s.stats["r0_all"] = round(1000 * (time.perf_counter() - _tA))
        s.stats["ms_e2"] += 1000 * (time.perf_counter() - t0); s.stats["n_render"] = s.stats.get("n_render", 0) + 1
        if "ms_r0" not in s.stats: s.stats["ms_r0"] = round(1000 * (time.perf_counter() - t0)); s.stats["r0_frames"] = len(s.frames)
        s._out(w_new, done_all); return True
    def _out(s, w, last):
        if not s.started:
            w = np.concatenate([s.pending, w]); idx = np.nonzero(np.abs(w) > 0.02)[0]
            if len(idx) == 0:
                s.pending = w[-PRE:]
                if not last: return
                w = w[:0]
            else:
                w = w[max(0, idx[0] - PRE):].copy(); s.started = True
                if os.environ.get('E2V2_FADEIN', '0') == '1': _k = min(len(w), 360); w[:_k] *= np.linspace(0, 1, _k)
        if len(w):
            if s.stats["t_first_audio"] is None: s.stats["t_first_audio"] = round(time.perf_counter() - s.stats["t0"], 3)
            for i in range(0, len(w), 4800): s.emit((np.clip(w[i:i + 4800], -1, 1) * 32767).astype(np.int16).tobytes())
    def _render_loop(s):
        try:
            while not s.cancelled:
                if not s._render(): s.rev.wait(0.005); s.rev.clear()
                if s.done and s.mel_out >= len(s.frames) * 8 and _pend0(s): break
        except Exception as e: print("RENDER ERROR", repr(e)[:300], flush=True); s.cancelled = True
    def run(s):
        if SCHED:
            s._done_ev = threading.Event(); s.c.engine.add(s); s._done_ev.wait(); return
        busy = False; s.rev = threading.Event(); s.par_on = False; rt = None
        try:
            while not s.cancelled:
                if not busy: s.ev.wait(0.01)
                s.ev.clear()
                if s.cancelled: break
                s._commit(); busy = bool(s._steps(6 if rt else (int(os.environ.get("T32_SERIAL_BUDGET", "6")) if not s.started else 2)))
                if rt: s.rev.set()
                else:
                    busy = busy | bool(s._render())
                    ahead = s.speech_off / 24000 - (time.perf_counter() - s.stats["t0"] - (s.stats.get("t_first_audio") or 1e9))
                    if PAR_E2 and s.mel_out > 0 and not s.done and (os.environ.get("T32_PAR_EARLY", "0") == "1" or (s.stats.get("t_first_audio") is not None and ahead >= PAR_AHEAD)):
                        s.par_on = True; rt = threading.Thread(target=s._render_loop, daemon=True); rt.start()
                if s.done and (rt or (s.mel_out >= len(s.frames) * 8 and _pend0(s))): break
            if rt: rt.join()
        finally:
            s.finished = True
            for k in ("ms_ctx", "ms_talker", "ms_e2"): s.stats[k] = round(s.stats[k])
            s.stats["sec"] = round(s.speech_off / 24000, 2); s.stats.pop("t0", None)
    def say_all(s, text):
        s.add_text(text); s.finish(); out = []; s.emit = out.append; s.run(); return np.frombuffer(b"".join(out), np.int16).astype(np.float32) / 32767
