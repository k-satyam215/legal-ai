"""
chat_engine.py v4 — Professional tone. Memory-aware. Follow-up engine.
Structured response display: LAW → ANALYSIS → STEPS → RISK → FOLLOW-UPS
"""
import re, logging
from backend.core.llm import call_llm
from backend.core.prompts import CHAT_SYSTEM, CHAT_USER
from backend.v2.memory import format_history_for_llm, is_followup, get_memory

logger = logging.getLogger(__name__)

_QUICK_CARDS: dict[str, str] = {
    "phone_lost": """📜 LAW: CrPC Section 154 (FIR), DoT CEIR Portal

🧠 ANALYSIS: Loss of a mobile phone requires an FIR to be filed under CrPC Section 154, which enables tracing and blocking of the device via IMEI. Simultaneously, the DoT's CEIR portal allows IMEI blocking to prevent misuse.

📌 STEPS:
1. File FIR at the nearest police station under CrPC Section 154
2. Block IMEI at ceir.gov.in (Department of Telecommunications)
3. Contact your carrier to block the SIM card

⚠️ RISK: LOW — property loss without criminal element

💬 YOU MAY ASK:
- What documents are required to file an FIR?
- Can I track my phone after filing the FIR?""",

    "phone_stolen": """📜 LAW: IPC Section 379 / BNS Section 303, CrPC Section 154

🧠 ANALYSIS: Mobile phone theft is a cognizable offence under IPC Section 379 (theft), requiring mandatory FIR registration under CrPC Section 154. Police are legally bound to register the FIR and investigate.

📌 STEPS:
1. File FIR immediately — IPC Section 379 / BNS Section 303
2. Block IMEI: ceir.gov.in and contact carrier for SIM block
3. Change all banking and account passwords immediately

⚠️ RISK: MEDIUM — financial loss risk if banking apps were accessed

💬 YOU MAY ASK:
- What if police refuse to register the FIR?
- How do I claim insurance for the stolen phone?""",

    "deposit_refund": """📜 LAW: Transfer of Property Act 1882, Section 108

🧠 ANALYSIS: Under Section 108 of the Transfer of Property Act 1882, a landlord is legally obligated to return the security deposit upon vacation of premises, subject to legitimate deductions. Failure to do so constitutes a breach of tenancy rights.

📌 STEPS:
1. Send written demand via WhatsApp and email (creates documentary evidence)
2. Issue Registered AD legal notice citing TPA Section 108
3. File recovery suit in Rent Control Court or Civil Court

⚠️ RISK: MEDIUM — prolonged litigation if no written agreement

💬 YOU MAY ASK:
- What deductions can the landlord legally make from the deposit?
- How do I draft a legal notice for deposit refund?""",

    "consumer_complaint": """📜 LAW: Consumer Protection Act 2019, Section 35

🧠 ANALYSIS: Under Section 35 of the Consumer Protection Act 2019, any consumer can file a complaint before the District Consumer Disputes Redressal Commission (DCDRC) for deficiency of service or defective goods. The complainant is entitled to refund, replacement, and compensation.

📌 STEPS:
1. File written complaint with company's grievance officer (mandatory first step)
2. Wait 30 days — if unresolved, file at consumerhelpline.gov.in
3. File complaint at DCDRC for claims up to ₹50 lakhs

⚠️ RISK: LOW — strong consumer protection legislation

💬 YOU MAY ASK:
- What documents are required for the consumer complaint?
- What compensation can I claim apart from refund?""",

    "salary_unpaid": """📜 LAW: Payment of Wages Act 1936, Section 15; Industrial Disputes Act 1947

🧠 ANALYSIS: Under Section 15 of the Payment of Wages Act 1936, an employee can file a claim before the Payment of Wages Authority for unpaid wages. Additionally, under the Industrial Disputes Act 1947, a complaint to the Labour Commissioner can initiate departmental action against the employer.

📌 STEPS:
1. Send formal written demand to HR and management via email
2. File complaint with Labour Commissioner (state-specific office)
3. File claim under Payment of Wages Act Section 15 before the authority

⚠️ RISK: MEDIUM — depends on employment classification and documentation

💬 YOU MAY ASK:
- Am I eligible if I was employed on contract basis?
- What interest or penalty can I claim on delayed wages?""",

    "fir_refused": """📜 LAW: CrPC Section 154, Section 156(3)

🧠 ANALYSIS: Under CrPC Section 154, registration of FIR for cognizable offences is mandatory and police cannot refuse. If refused, Section 156(3) CrPC enables a Magistrate to direct investigation, and a High Court Writ Petition can be filed for appropriate orders.

📌 STEPS:
1. Submit written complaint to SP/DCP of the district
2. File application before Magistrate under CrPC Section 156(3)
3. File online complaint at cybercrime.gov.in for cyber-related matters

⚠️ RISK: HIGH — delay in FIR can affect evidence and investigation

💬 YOU MAY ASK:
- What is the procedure for filing a Section 156(3) application?
- Can I file an FIR at any police station or only the local one?""",

    "cyber_fraud": """📜 LAW: IT Act 2000 Section 66D; IPC Section 420 / BNS Section 318

🧠 ANALYSIS: Cyber fraud involving financial loss is punishable under IT Act Section 66D (cheating by impersonation online) and IPC Section 420 / BNS Section 318 (cheating). Immediate reporting to cybercrime.gov.in and the bank is critical for fund recovery.

📌 STEPS:
1. Report immediately at cybercrime.gov.in (National Cybercrime Reporting Portal)
2. Call your bank immediately — request transaction hold or reversal
3. File FIR at local police station citing IT Act Section 66D

⚠️ RISK: HIGH — funds recovery depends on speed of reporting

💬 YOU MAY ASK:
- What evidence should I preserve for the cyber fraud complaint?
- Is there a time limit to report and recover funds?""",

    "eviction": """📜 LAW: Transfer of Property Act 1882, Section 108; IPC Section 441

🧠 ANALYSIS: Under TPA Section 108, a landlord cannot evict a tenant without following due legal process, and any forceful eviction or disconnection of utilities constitutes an offence under IPC Section 441 (criminal trespass). Tenants have a right to approach the Rent Control Court.

📌 STEPS:
1. File police complaint — IPC Section 441 for criminal trespass
2. File urgent application at Rent Control Court for stay of eviction
3. Document all communication and threats as evidence

⚠️ RISK: HIGH — immediate action required to prevent illegal eviction

💬 YOU MAY ASK:
- What is the legal eviction procedure a landlord must follow?
- Can I claim damages for illegal eviction attempt?""",

    "documents": """📄 REQUIRED LEGAL DOCUMENTS BY CASE TYPE:

For FIR / Criminal cases:
• FIR copy (free from police station)
• Complaint letter, evidence screenshots, witness details

For Consumer complaints:
• Purchase bill / invoice, warranty card
• Communication screenshots, complaint reference numbers

For Employment disputes:
• Appointment letter, payslips, bank statements
• HR communications, termination letter if any

For Rent disputes:
• Rent agreement, deposit receipt, move-out photos
• WhatsApp/email communication with landlord

General legal documents:
• Affidavit (Notary — ₹50-200)
• Legal Notice (advocate or self-drafted)
• RTI Application (₹10 — for government information)""",
}

_INTENTS: list[tuple[str, list[str]]] = [
    ("phone_stolen",    ["phone chori","mobile chori","chheen liya","snatched","phone stolen"]),
    ("phone_lost",      ["phone kho","mobile kho","phone lost","phone nahi mila","gum ho"]),
    ("deposit_refund",  ["deposit nahi","security deposit","kiraya wapas","deposit return"]),
    ("consumer_complaint", ["consumer complaint","refund nahi","defective","amazon nahi","flipkart nahi","wrong product"]),
    ("salary_unpaid",   ["salary nahi","payment nahi","wages nahi","paisa nahi diya","salary due"]),
    ("fir_refused",     ["fir nahi","police nahi sun","complaint nahi li","fir refuse"]),
    ("eviction",        ["evict","ghar se nikal","bijli band","khali karo","forceful eviction"]),
    ("cyber_fraud",     ["online fraud","cyber fraud","upi fraud","bank se paisa","otp diya","online scam"]),
    ("documents",       ["kaunse documents","papers chahiye","kya proof","documents list"]),
]

_DEEP_RE = re.compile(
    r"(section|law|act|ipc|crpc|bnss|legal notice|court|sue|rights kya|"
    r"compensation|damages|magistrate|advocate|vakil|high court|procedure|kab tak)", re.I)


def _detect_intent(msg: str) -> str | None:
    m = msg.lower()
    for intent, patterns in _INTENTS:
        if any(p in m for p in patterns): return intent
    return None


def chat_response(message: str, history: list[dict],
                  case_type: str = "general", session_id: str = "default") -> dict:
    try:
        # 1. Quick card — instant, structured, professional
        intent = _detect_intent(message)
        if intent and intent in _QUICK_CARDS:
            return {
                "reply": _QUICK_CARDS[intent],
                "quick_card": intent,
                "needs_deep_advice": True,
                "suggested_action": "For detailed legal analysis, use the 'Deep Analysis' tab.",
                "detected_intent": intent,
            }

        # 2. Format history
        hist_text = format_history_for_llm(history, max_turns=4)
        followup  = is_followup(message, history)

        # 3. Inject case state context for follow-ups
        mem = get_memory()
        ctx = mem.get_context(session_id)
        case_state_ctx = ctx.get("case_state","")
        if case_state_ctx and followup:
            enhanced_msg = f"[Context: {case_state_ctx}]\n\nQuestion: {message}"
        else:
            enhanced_msg = message

        # 4. LLM call
        raw = call_llm(
            messages=[
                {"role":"system","content":CHAT_SYSTEM},
                {"role":"user","content":CHAT_USER.format(history=hist_text, message=enhanced_msg)},
            ],
            temperature=0.15, max_tokens=160,
        )

        reply      = raw.strip()
        needs_deep = bool(_DEEP_RE.search(message)) or (len(message.split()) > 8 and not followup)

        # 5. Store in memory
        mem.add_turn(session_id, "user", message)
        mem.add_turn(session_id, "assistant", reply)

        return {
            "reply": reply,
            "quick_card": None,
            "needs_deep_advice": needs_deep,
            "suggested_action": "For detailed legal analysis with law sections, use the 'Deep Analysis' tab." if needs_deep else None,
            "detected_intent": None,
        }
    except Exception as e:
        logger.error(f"[Chat] {e}")
        return {"reply":"A technical error occurred. Please try again.",
                "quick_card":None,"needs_deep_advice":False,"suggested_action":None,"detected_intent":None}
