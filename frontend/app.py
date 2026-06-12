"""AI Legal Advisor India — Streamlit Frontend v3"""
import os, sys, tempfile, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st

st.set_page_config(page_title="AI Legal Advisor — India", page_icon="⚖️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0f1117;color:#e2e8f0;}
section[data-testid="stSidebar"]{background:#1a1d27!important;border-right:1px solid #2d3748;}
.card{background:#1a1d27;border:1px solid #2d3748;border-radius:12px;padding:18px;margin:8px 0;}
.deep-card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:12px;padding:18px;margin:8px 0;}
.risk-high{color:#fc8181;font-weight:700;} .risk-medium{color:#f6ad55;font-weight:700;} .risk-low{color:#68d391;font-weight:700;}
.step{background:#1e2535;border-left:3px solid #4299e1;padding:10px 14px;margin:5px 0;border-radius:5px;font-size:.91em;}
.step-deep{background:#1a2535;border-left:3px solid #805ad5;padding:10px 14px;margin:5px 0;border-radius:5px;font-size:.91em;}
.law-chip{display:inline-block;background:#1a365d;border:1px solid #2b6cb0;border-radius:20px;padding:3px 11px;font-size:.82em;margin:3px;color:#90cdf4;}
.law-alt{display:inline-block;background:#1a2835;border:1px solid #553c9a;border-radius:20px;padding:3px 11px;font-size:.82em;margin:3px;color:#b794f4;}
.chat-u{background:#2b4c7e;border-radius:16px 16px 4px 16px;padding:10px 14px;margin:6px 0 6px auto;max-width:80%;float:right;clear:both;font-size:.91em;}
.chat-b{background:#1e2535;border:1px solid #2d3748;border-radius:16px 16px 16px 4px;padding:10px 14px;margin:6px auto 6px 0;max-width:85%;float:left;clear:both;font-size:.91em;line-height:1.6;}
.quick{background:#1a2e1a;border:1px solid #276327;border-radius:10px;padding:14px;margin:6px 0;white-space:pre-line;font-size:.88em;line-height:1.7;float:left;clear:both;max-width:90%;}
.clr{clear:both;height:4px;}
.issue{background:#1a365d;border:1px solid #2b6cb0;border-radius:8px;padding:8px 14px;font-size:.88em;color:#bee3f8;margin:6px 0;display:block;}
.strategy{background:#1a2e1a;border:1px solid #2f855a;border-radius:8px;padding:8px 14px;font-size:.88em;color:#c6f6d5;margin:6px 0;display:block;}
.best{background:#0f2016;border:1px solid #276327;border-radius:8px;padding:12px;margin:4px 0;font-size:.88em;}
.worst{background:#200f0f;border:1px solid #c53030;border-radius:8px;padding:12px;margin:4px 0;font-size:.88em;}
.edge{background:#1a1a2e;border:1px solid #553c9a;border-radius:8px;padding:12px;margin:4px 0;font-size:.88em;}
.fq{background:#1a1d27;border:1px solid #4a5568;border-radius:8px;padding:8px 14px;margin:4px 0;font-size:.85em;color:#a0aec0;}
.stButton>button{background:#2b6cb0;color:white;border:none;border-radius:8px;padding:9px 22px;font-weight:600;}
.stButton>button:hover{background:#2c5282;}
.stTextInput>div>div>input,.stTextArea textarea{background:#1a1d27!important;border:1px solid #2d3748!important;color:#e2e8f0!important;border-radius:8px!important;}
div[data-testid="stChatInput"]>div{background:#1a1d27!important;border:1px solid #2d3748!important;border-radius:12px!important;}
.stTabs [data-baseweb="tab-list"]{background:#1a1d27;border-radius:8px;gap:3px;}
.stTabs [data-baseweb="tab"]{color:#a0aec0;border-radius:6px;padding:7px 18px;}
.stTabs [aria-selected="true"]{background:#2b6cb0!important;color:white!important;}
hr{border-color:#2d3748!important;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⚖️ Loading AI Legal Advisor...")
def load_backend():
    from backend.rag.loader import initialize
    initialize()
    from backend.v2.router_v2 import route_query_v2, route_deep_analysis
    from backend.v2.chat_engine import chat_response
    from backend.services.notice_generator import generate_notice_text, generate_notice_pdf
    from backend.services.timeline_generator import generate_timeline
    return route_query_v2, route_deep_analysis, chat_response, generate_notice_text, generate_notice_pdf, generate_timeline

try:
    route_query_v2, route_deep_analysis, chat_response, gen_notice_text, gen_notice_pdf, gen_timeline = load_backend()
    READY = True
except Exception as e:
    READY = False; ERR = str(e)

for k, v in [("chat_hist",[]),("last_adv",None),("last_deep",None),("sid",str(uuid.uuid4())[:12])]:
    if k not in st.session_state: st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:16px 0 8px">
    <div style="font-size:2.2em">⚖️</div>
    <div style="font-size:1.05em;font-weight:700;color:#90cdf4">AI Legal Advisor</div>
    <div style="font-size:.73em;color:#718096">India • LLaMA-3.3-70b + RAG v3</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("**💬 Quick Examples**")
    for ex in ["Mera phone kho gaya","Landlord deposit nahi de raha","Salary 3 mahine se nahi mili","Cyber fraud ho gaya","Boss ne bina notice fired kiya","FIR kaise karein"]:
        if st.button(ex, key=f"sb_{ex}", use_container_width=True):
            st.session_state.chat_hist.append({"role":"user","content":ex})
            st.session_state._pend = ex
    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Chat",use_container_width=True): st.session_state.chat_hist=[]; st.rerun()
    with c2:
        if st.button("🗑️ Q&A",use_container_width=True): st.session_state.last_adv=None; st.session_state.last_deep=None; st.rerun()
    st.divider()
    st.markdown("<div style='font-size:.7em;color:#4a5568;text-align:center'>⚠️ Informational only.</div>",unsafe_allow_html=True)

st.markdown("<h1 style='margin-bottom:2px;color:#e2e8f0'>⚖️ AI Legal Advisor — India</h1><p style='color:#718096;margin-top:0;font-size:.88em'>Indian Law + RAG + LLaMA-3.3-70b • v3</p>",unsafe_allow_html=True)

if not READY: st.error(f"⚠️ {ERR}"); st.stop()

tab_chat, tab_ask, tab_deep, tab_notice, tab_tl = st.tabs(["💬 Chat","🔍 Quick Analysis","🧠 Deep Analysis","📄 Notice","📅 Timeline"])

# ══ TAB 1: CHAT ═══════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("""<div style='background:#1a2535;border:1px solid #2d3748;border-radius:8px;
    padding:10px 14px;margin-bottom:12px;font-size:.84em;color:#90cdf4'>
    💡 Koi bhi legal problem batao — phone chori, deposit, salary, FIR, fraud sab handle hota hai
    </div>""", unsafe_allow_html=True)

    if not st.session_state.chat_hist:
        st.markdown("<div style='text-align:center;padding:36px;color:#4a5568'><div style='font-size:1.8em'>💬</div><div>Apna legal issue yahan likho...</div></div>",unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_hist:
            if msg["role"]=="user":
                st.markdown(f'<div class="chat-u">👤 {msg["content"]}</div><div class="clr"></div>',unsafe_allow_html=True)
            else:
                css = "quick" if msg.get("qc") else "chat-b"
                ic  = "📋" if msg.get("qc") else "⚖️"
                st.markdown(f'<div class="{css}">{ic} {msg["content"]}</div><div class="clr"></div>',unsafe_allow_html=True)
                if msg.get("nd"):
                    st.markdown("<div style='clear:both;margin:0 0 4px 6px;font-size:.77em;color:#4299e1'>💡 Detailed analysis ke liye 'Deep Analysis' tab use karein</div>",unsafe_allow_html=True)

    pend = getattr(st.session_state,"_pend",None)
    if pend:
        del st.session_state._pend
        hist_api = [{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_hist[:-1]]
        with st.spinner("🤔 ..."):
            res = chat_response(pend, hist_api, session_id=st.session_state.sid)
        st.session_state.chat_hist.append({"role":"assistant","content":res["reply"],"qc":res.get("quick_card"),"nd":res.get("needs_deep_advice")})
        st.rerun()

    inp = st.chat_input("Apna legal issue likho...")
    if inp:
        inp = inp.strip()
        st.session_state.chat_hist.append({"role":"user","content":inp})
        hist_api = [{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_hist[:-1]]
        with st.spinner("🤔 ..."):
            res = chat_response(inp, hist_api, session_id=st.session_state.sid)
        st.session_state.chat_hist.append({"role":"assistant","content":res["reply"],"qc":res.get("quick_card"),"nd":res.get("needs_deep_advice")})
        st.rerun()

# ══ TAB 2: QUICK ANALYSIS ════════════════════════════════════════════════════
with tab_ask:
    st.markdown("""<div style='background:#1a2535;border:1px solid #2d3748;border-radius:8px;
    padding:10px 14px;margin-bottom:12px;font-size:.84em;color:#90cdf4'>
    🔍 Fast legal analysis — exact laws, steps, strategy &lt;1200ms
    </div>""", unsafe_allow_html=True)

    if st.session_state.last_adv:
        a = st.session_state.last_adv
        risk = a.get("risk_level","medium")
        ri = {"low":"🟢","medium":"🟡","high":"🔴"}.get(risk,"🟡")
        st.markdown(f'<span class="issue">⚠️ {a.get("issue","")}</span>',unsafe_allow_html=True)
        st.markdown(f'<span class="strategy">🎯 {a.get("strategy","")}</span>',unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Case Type", a.get("case_type","").upper())
        with c2: st.metric("Risk", f"{ri} {risk.upper()}")
        with c3: st.metric("Notice", "✅" if a.get("notice_applicable") else "❌")
        if a.get("laws"):
            st.markdown("**⚖️ Laws**")
            st.markdown(" ".join(f'<span class="law-chip">{l}</span>' for l in a["laws"]),unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="line-height:1.7">{a.get("analysis","")}</div>',unsafe_allow_html=True)
        st.markdown("**📌 Steps**")
        for i,s in enumerate(a.get("steps",[]),1):
            st.markdown(f'<div class="step"><b>{i}.</b> {s}</div>',unsafe_allow_html=True)
        if a.get("follow_up_questions"):
            st.markdown("**❓ Follow-up**")
            for q in a["follow_up_questions"]:
                st.markdown(f'<div class="fq">💭 {q}</div>',unsafe_allow_html=True)

    q = st.chat_input("Legal question likhein...", key="ask_inp")
    if q:
        with st.spinner("⚖️ Analyzing..."):
            st.session_state.last_adv = route_query_v2(q.strip())
        st.rerun()

# ══ TAB 3: DEEP ANALYSIS ════════════════════════════════════════════════════
with tab_deep:
    st.markdown("""<div style='background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;
    padding:10px 14px;margin-bottom:12px;font-size:.84em;color:#90cdf4'>
    🧠 Deep legal reasoning — edge cases, risks, alternatives, interpretation
    </div>""", unsafe_allow_html=True)

    dq  = st.text_area("Situation detail mein describe karein:", height=90,
                        placeholder="Example: Landlord ne 50000 deposit liya 2 saal pehle. Flat vacate kiya 3 mahine pehle. Koi written agreement nahi. Rent receipts hain mujhe.")
    ext = st.text_input("Extra context (optional):", placeholder="Amount: ₹50,000 | State: Delhi | Duration: 3 months")

    if st.button("🧠 Run Deep Analysis", type="primary", use_container_width=True):
        if dq.strip():
            with st.spinner("🧠 Deep analysis..."):
                st.session_state.last_deep = route_deep_analysis(dq.strip(), extra_context=ext)
            st.rerun()
        else: st.warning("Please describe your situation.")

    if st.session_state.last_deep:
        d    = st.session_state.last_deep
        risk = d.get("risk_level","medium")
        ri   = {"low":"🟢","medium":"🟡","high":"🔴"}.get(risk,"🟡")
        st.markdown(f'<span class="issue">⚠️ {d.get("issue","")}</span>',unsafe_allow_html=True)
        st.markdown(f'<span class="strategy">🎯 {d.get("strategy","")}</span>',unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("Case Type", d.get("case_type","").upper())
        with c2: st.metric("Risk", f"{ri} {risk.upper()}")
        with c3: st.metric("Timeline", d.get("timeline_estimate","N/A")[:20])
        st.markdown(f'<div class="deep-card"><div style="font-size:.75em;color:#718096">PRIMARY LAW</div><div style="color:#90cdf4;font-weight:600">{d.get("primary_law","")}</div></div>',unsafe_allow_html=True)
        if d.get("laws"):
            st.markdown(" ".join(f'<span class="law-chip">{l}</span>' for l in d["laws"]),unsafe_allow_html=True)
        st.markdown(f'<div class="deep-card"><div style="font-size:.75em;color:#718096">⚖️ LEGAL INTERPRETATION</div><div style="line-height:1.7;margin-top:6px">{d.get("legal_interpretation","")}</div></div>',unsafe_allow_html=True)
        sa = d.get("scenario_analysis",{})
        if sa:
            st.markdown("**📊 Scenario Analysis**")
            c1,c2 = st.columns(2)
            with c1: st.markdown(f'<div class="best"><div style="font-size:.75em;color:#68d391">✅ BEST CASE</div>{sa.get("best_case","")}</div>',unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="worst"><div style="font-size:.75em;color:#fc8181">⚠️ WORST CASE</div>{sa.get("worst_case","")}</div>',unsafe_allow_html=True)
            for ec in sa.get("edge_cases",[]):
                st.markdown(f'<div class="edge"><span style="font-size:.75em;color:#b794f4">🔀 EDGE: </span>{ec}</div>',unsafe_allow_html=True)
        st.markdown("**📌 Steps**")
        for i,s in enumerate(d.get("steps",[]),1):
            st.markdown(f'<div class="step-deep"><b>{i}.</b> {s}</div>',unsafe_allow_html=True)
        if d.get("alternative_remedies"):
            st.markdown("**🔄 Alternatives**")
            st.markdown(" ".join(f'<span class="law-alt">{a}</span>' for a in d["alternative_remedies"]),unsafe_allow_html=True)
        if d.get("risk_factors"):
            st.markdown("**⚠️ Risk Factors**")
            for rf in d["risk_factors"]:
                st.markdown(f'<div class="step" style="border-left-color:#fc8181">⚠️ {rf}</div>',unsafe_allow_html=True)
        if d.get("follow_up_questions"):
            st.markdown("**❓ Questions to Strengthen Case**")
            for q in d["follow_up_questions"]:
                st.markdown(f'<div class="fq">💭 {q}</div>',unsafe_allow_html=True)

# ══ TAB 4: NOTICE ═══════════════════════════════════════════════════════════
with tab_notice:
    st.markdown("### 📄 Legal Notice Generator")
    adv   = st.session_state.last_adv or st.session_state.last_deep
    _map  = {"rent":"deposit_refund","consumer":"consumer_complaint","employment":"employment_termination"}
    _typs = ["deposit_refund","eviction","consumer_complaint","employment_termination","general"]
    c1,c2 = st.columns(2)
    with c1:
        def_i   = _typs.index(_map.get(adv.get("case_type",""),"general")) if adv else 0
        ntype   = st.selectbox("Type", _typs, index=def_i)
        sname   = st.text_input("Your Name")
        saddr   = st.text_area("Your Address", height=70)
    with c2:
        rname   = st.text_input("Recipient Name")
        raddr   = st.text_area("Recipient Address", height=70)
        relief  = st.text_input("Relief (e.g. ₹50,000 refund)")
    facts   = st.text_area("Key Facts", height=85, placeholder="Vacated flat 1-Jan-2025. Deposit ₹50,000. Not returned despite 3 months.")
    laws_opts = (adv.get("laws",[]) if adv else []) or []
    law_ref = st.selectbox("Law", laws_opts + ["Custom..."]) if laws_opts else st.text_input("Applicable Law")
    if law_ref == "Custom...": law_ref = st.text_input("Enter law:", placeholder="Transfer of Property Act, 1882, Section 108")
    gpdf    = st.checkbox("Generate PDF")
    if st.button("⚡ Generate Notice", type="primary"):
        if not all([sname,saddr,rname,raddr,facts,relief]):
            st.warning("⚠️ All fields required.")
        else:
            with st.spinner("Drafting..."):
                try:
                    text = gen_notice_text(notice_type=ntype,sender_name=sname,sender_address=saddr,
                                           recipient_name=rname,recipient_address=raddr,
                                           facts=facts,relief=relief,law=law_ref or "applicable law")
                    editable = st.text_area("Edit:", value=text, height=400, key="ned")
                    if gpdf:
                        with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
                            pp = gen_notice_pdf(editable,sname,tmp.name)
                            with open(pp,"rb") as f: st.download_button("📥 PDF",f.read(),"legal_notice.pdf","application/pdf")
                    else: st.download_button("📥 TXT",editable,"legal_notice.txt","text/plain")
                except Exception as e: st.error(f"Error: {e}")

# ══ TAB 5: TIMELINE ══════════════════════════════════════════════════════════
with tab_tl:
    st.markdown("### 📅 Legal Action Timeline")
    adv    = st.session_state.last_adv or st.session_state.last_deep
    ctypes = ["rent","consumer","criminal","employment","general"]
    ttype  = st.selectbox("Case Type", ctypes, index=ctypes.index(adv.get("case_type","general")) if adv else 0)
    tfacts = st.text_area("Facts", value=adv.get("analysis","")[:180] if adv else "", height=70)
    tout   = st.text_input("Desired Outcome", placeholder="Full deposit refund + 9% interest")
    if st.button("📅 Generate Timeline", type="primary"):
        if not tfacts or not tout: st.warning("Fill facts and outcome.")
        else:
            with st.spinner("Generating..."):
                try:
                    ms = gen_timeline(ttype, tfacts, tout)
                    for m in ms:
                        badge = "🔴" if m.get("is_critical") else "⚪"
                        with st.expander(f"{badge} Day {m.get('estimated_days_from_start','?')} — {m.get('phase','')} → {m.get('title','')}"):
                            st.write(m.get("description",""))
                            if m.get("escalation_path"): st.markdown(f"**🔀 If fails:** {m['escalation_path']}")
                    import pandas as pd
                    df = pd.DataFrame([{"Step":m.get("title","")[:28],"Day":m.get("estimated_days_from_start",0)} for m in ms])
                    if not df.empty: st.bar_chart(df.set_index("Step")["Day"])
                except Exception as e: st.error(f"Error: {e}")
