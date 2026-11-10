"""
Demonstration and Validation Suite for the Phishing Email Classification System.
Runs test scenarios across various phishing lures and legitimate workplace communications.
"""

from predict import analyze_email, print_analysis

DEMO_EMAILS = [
    {
        "category": "Phishing - Urgent Account Suspension Lure",
        "expected": "Phishing",
        "text": """
        URGENT SECURITY NOTICE: Your PayPal account has been temporarily restricted due to suspicious login attempts.
        Please verify your identity immediately by clicking on the secure link below to avoid permanent suspension.
        Click here: http://secure-paypal-verification-update.com/login?id=83921
        Failure to verify within 24 hours will result in account closure.
        """
    },
    {
        "category": "Phishing - Nigerian Advance Fee Fraud / Inheritance",
        "expected": "Phishing",
        "text": """
        DEAR BELOVED FRIEND, I am Barrister Mohammed Musa, attorney to late Dr. George who died with his family in a plane crash.
        He left behind an unclaimed sum of $25.5 Million USD in a security escrow vault.
        I am seeking your urgent assistance to stand as foreign beneficiary so the funds can be transferred to your account.
        You will receive 40% of the total funds. Please reply with your full legal name, private telephone number, and bank details.
        """
    },
    {
        "category": "Phishing - Fake IT Department Credential Harvesting",
        "expected": "Phishing",
        "text": """
        ATTENTION ALL EMPLOYEES: IT Helpdesk scheduled server migration.
        Your email storage quota has exceeded the 99% threshold.
        To maintain access to your corporate email and prevent account deletion, please update your login password immediately:
        http://it-support-portal-webmail-update.org/auth
        IT Support Desk & System Administration
        """
    },
    {
        "category": "Legitimate - Workplace Meeting Schedule",
        "expected": "Legitimate",
        "text": """
        Hi Sarah,
        Could we reschedule our bi-weekly architecture sync from Thursday morning to Friday at 2:00 PM?
        I need to finalize the quarterly performance metrics and review the API latency logs before we meet.
        Let me know if that time works for you.
        Best regards,
        David Chen
        Software Engineering Lead
        """
    },
    {
        "category": "Legitimate - Open Source Pull Request & Code Review",
        "expected": "Legitimate",
        "text": """
        Thanks for submitting pull request #142!
        I reviewed the commit diff for the database connection pooling fix.
        Overall the logic looks clean, but could you add a unit test covering timeout retries under network partition?
        Once the CI tests pass on Linux and Windows, I'll approve and merge.
        Cheers,
        Alex
        """
    },
    {
        "category": "Legitimate - Standard Business Invoice / PO Confirmation",
        "expected": "Legitimate",
        "text": """
        Good morning Robert,
        Attached is the approved purchase order #88412 for the cloud infrastructure renewal.
        Please confirm receipt and let us know the estimated provisioning timeline.
        Thank you for your partnership.
        Accounts Payable Department
        """
    }
]


def run_demo():
    print("=" * 70)
    print("   RUNNING AUTOMATED TEST CASES & DEMO")
    print("=" * 70)

    correct_count = 0
    total = len(DEMO_EMAILS)

    for i, test_case in enumerate(DEMO_EMAILS, 1):
        print(f"\n--- Scenario {i}/{total}: {test_case['category']} ---")
        res = analyze_email(test_case["text"])
        print_analysis(res, text_snippet=test_case["text"])

        is_match = res["prediction"] == test_case["expected"]
        if is_match:
            correct_count += 1
            print(f" [PASS] Expected: {test_case['expected']} | Actual: {res['prediction']}\n")
        else:
            print(f" [FAIL] Expected: {test_case['expected']} | Actual: {res['prediction']}\n")

    accuracy = (correct_count / total) * 100
    print("=" * 70)
    print(f" Demo Test Results: {correct_count}/{total} Passed ({accuracy:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
