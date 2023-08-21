QA_ANSWER_SYSTEM_PROMPT = (
    "You are Paul Graham, a famous entrepreneur and startup founder and investor, as well as one of the authors of the "
    "lisp programming language. You love being helpful and answering questions about your essays."
)

QA_REDUCE_PROMPT = """
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Your answer should be self contained and not make any references to any provided documents. It's okay to make references to the question itself.
ALWAYS include a "SOURCES" part in your answer.

QUESTION: How can bananas improve my health?
=========
Content: A banana is an elongated, edible fruit – botanically a berry – produced by several kinds of large herbaceous flowering plants in the genus Musa. In some countries, bananas used for cooking may be called "plantains", distinguishing them from dessert bananas. The fruit is variable in size, color, and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a rind, which may be green, yellow, red, purple, or brown when ripe.
Source: 4-pl
Content: The fruits grow upward in clusters near the top of the plant. Almost all modern edible seedless (parthenocarp) bananas come from two wild species – Musa acuminata and Musa balbisiana. The scientific names of most cultivated bananas are Musa acuminata, Musa balbisiana, and Musa × paradisiaca for the hybrid Musa acuminata × M. balbisiana, depending on their genomic constitution. The old scientific name for this hybrid, Musa sapientum, is no longer used.
Source: 7-pl
=========
FINAL ANSWER: I don't know.
SOURCES:

QUESTION: What is an EMI option scheme?
=========
Content: "In a nutshell, the EMI option scheme is the most tax-efficient way to grant options to your employees. The EMI, which stands for Enterprise Management Incentive, is a share option scheme backed by HMRC in the UK."
Source: 6-pl
Content: "The Enterprise Management Incentive option scheme (EMI for short) is an HMRC initiative that allows UK businesses to give share options to their employees."
Source: 9-pl
Content: "First, we have the extremely popular EMI Option Scheme, designed to incentivise and reward employees by offering them a right to get shares in the company if they work for you long enough (time-based vesting) or if they help you achieve certain business goals (milestone vesting)."
Source: 33-pl
Content: "An EMI Option Scheme allows you to grant options to qualifying employees worth up to £250k (each employee) without giving rise to an income tax or NIC charge."
Source: 46-pl
Content: "What is an EMI scheme."
Source: 17-pl
Content: "An EMI Options Scheme is a tax-advantaged share options scheme provided by HMRC that allows qualifying companies to give share options to employees in a way that avoids the company incurring a tax liability, and greatly reducing the tax liability for the employee."
Source: 18-pl
Content: "Crucially, to reap the benefits of an EMI Option Scheme the option holder must be an employee of the company or the holding company of the group. Consultants, advisers and non-executive directors are all excluded."
Source: 13-pl
Content: "an EMI Scheme"
Source: 11-pl
Content: "EMI share options explained"
Source: 12-pl
=========
FINAL ANSWER: An EMI (Enterprise Management Incentive) option scheme is designed to incentivize and reward employees by offering them the right to get shares based on pre-defined conditions such as time-based vesting or milestone-based vesting. This scheme offers the most tax advantageous way for the company to grant options to employees.
SOURCES: 6-pl, 9-pl, 33-pl, 46-pl, 18-pl

QUESTION: {question}
=========
{excerpts}
=========
FINAL ANSWER:
""".strip()
