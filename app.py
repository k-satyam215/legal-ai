import streamlit as st
from backend.rag.qa_system import ask_question
from backend.notice_generator import generate_notice

st.set_page_config(page_title="AI Legal Advisor", layout="centered")

st.title("⚖️ AI Legal Advisor (India)")
st.write("Ask any legal question and get guidance + notice generation")

query = st.text_input("Enter your legal issue")

if st.button("Get Legal Help"):

    if query:
        data = ask_question(query)

        if isinstance(data, dict):

            st.subheader("📜 Law")
            st.write(data["law"])

            st.subheader("🧠 Explanation")
            st.write(data["explanation"])

            st.subheader("📌 Steps")
            for step in data["steps"]:
                st.write("•", step)

            # 🔥 CONDITIONAL NOTICE UI
            if data.get("case_type") == "rent" and data.get("notice_points"):

                st.divider()
                st.subheader("📄 Generate Legal Notice")

                name = st.text_input("Your Name")
                address = st.text_input("Your Address")
                landlord_name = st.text_input("Landlord Name")
                landlord_address = st.text_input("Landlord Address")
                amount = st.text_input("Deposit Amount")

                if st.button("Generate Notice PDF"):

                    user_info = {
                        "name": name,
                        "address": address,
                        "landlord_name": landlord_name,
                        "landlord_address": landlord_address,
                        "amount": amount
                    }

                    generate_notice(data, user_info)

                    st.success("✅ Legal Notice Generated!")

                    with open("legal_notice.pdf", "rb") as f:
                        st.download_button(
                            "📥 Download PDF",
                            f,
                            file_name="legal_notice.pdf"
                        )

            else:
                st.info("ℹ️ Legal notice not applicable for this case.")

        else:
            st.error("❌ Something went wrong")