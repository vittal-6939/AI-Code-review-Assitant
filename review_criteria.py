"""
review_criteria.py

A curated taxonomy of labeled Python code-review examples used to ground
the AI code reviewer's judgment. Instead of relying purely on the LLM's
implicit notion of "bug" / "security issue" / "style issue", these examples
are embedded into a separate Chroma vectorstore (`review_criteria` collection)
and retrieved per code chunk at review time, so the model has concrete
precedent to pattern-match the uploaded code against.

Usage (in app.py):

    from review_criteria import REVIEW_CRITERIA, build_criteria_documents
    from langchain_chroma import Chroma

    criteria_docs = build_criteria_documents(REVIEW_CRITERIA)
    criteria_vectorstore = Chroma.from_documents(
        documents=criteria_docs,
        embedding=embeddings,
    )

    # Per chunk of uploaded code:
    relevant_criteria = criteria_vectorstore.similarity_search(chunk.page_content, k=3)
"""

from dataclasses import dataclass, field
from typing import List
from langchain_core.documents import Document


@dataclass
class CriteriaExample:
    category: str          # "Bug" | "Security" | "Performance" | "Style"
    subcategory: str       # short label, e.g. "Mutable Default Argument"
    severity: str          # "High" | "Med" | "Low"
    bad_pattern: str       # illustrative snippet showing the issue
    why_flagged: str       # explanation of the underlying problem
    good_pattern: str      # a corrected/idiomatic version
    tags: List[str] = field(default_factory=list)  # optional keywords for search


# ---------------------------------------------------------------------------
# 1. BUGS — correctness issues that produce wrong behavior, not just bad style
# ---------------------------------------------------------------------------

BUGS: List[CriteriaExample] = [
    CriteriaExample(
        category="Bug",
        subcategory="Mutable Default Argument",
        severity="High",
        bad_pattern='''
def add_item(item, cache=[]):
    cache.append(item)
    return cache
''',
        why_flagged=(
            "Default argument objects are created once, at function definition "
            "time, and reused across all calls. Every call that omits `cache` "
            "shares and mutates the same list, causing state to leak between "
            "unrelated calls."
        ),
        good_pattern='''
def add_item(item, cache=None):
    if cache is None:
        cache = []
    cache.append(item)
    return cache
''',
        tags=["mutable default", "function argument", "shared state"],
    ),
    CriteriaExample(
        category="Bug",
        subcategory="Bare Except Swallowing Errors",
        severity="High",
        bad_pattern='''
try:
    process(record)
except:
    pass
''',
        why_flagged=(
            "A bare `except:` catches everything, including KeyboardInterrupt "
            "and SystemExit, and silently discards the error. Real bugs "
            "(TypeErrors, missing keys, etc.) disappear without a trace, "
            "making the failure impossible to diagnose later."
        ),
        good_pattern='''
try:
    process(record)
except ValueError as e:
    logger.warning("Skipping malformed record: %s", e)
''',
        tags=["exception handling", "silent failure", "error swallowing"],
    ),
    CriteriaExample(
        category="Bug",
        subcategory="Floating Point Equality",
        severity="Med",
        bad_pattern='''
total = 0.1 + 0.2
if total == 0.3:
    print("exact match")
''',
        why_flagged=(
            "Floating point numbers cannot represent most decimal values "
            "exactly, so `0.1 + 0.2 == 0.3` evaluates to False due to "
            "rounding error. Direct equality comparisons on floats are "
            "almost always a latent bug."
        ),
        good_pattern='''
import math
if math.isclose(total, 0.3, rel_tol=1e-9):
    print("close enough")
''',
        tags=["floating point", "equality", "numeric precision"],
    ),
    CriteriaExample(
        category="Bug",
        subcategory="Identity Comparison Instead of Equality",
        severity="Med",
        bad_pattern='''
if user_input is "yes":
    proceed()
''',
        why_flagged=(
            "`is` checks object identity, not value equality. For strings "
            "this can appear to work due to CPython's string interning of "
            "short literals, but it is not guaranteed behavior and breaks "
            "for longer or dynamically constructed strings."
        ),
        good_pattern='''
if user_input == "yes":
    proceed()
''',
        tags=["is vs ==", "identity", "string comparison"],
    ),
    CriteriaExample(
        category="Bug",
        subcategory="Mutating a List While Iterating",
        severity="High",
        bad_pattern='''
for item in items:
    if item.expired:
        items.remove(item)
''',
        why_flagged=(
            "Removing elements from a list while iterating over it shifts "
            "indices mid-loop, causing elements to be silently skipped. "
            "This is a common source of intermittent, hard-to-reproduce bugs."
        ),
        good_pattern='''
items[:] = [item for item in items if not item.expired]
''',
        tags=["iteration", "list mutation", "skipped elements"],
    ),
    CriteriaExample(
        category="Bug",
        subcategory="Off-by-One in Range Bound",
        severity="Med",
        bad_pattern='''
for i in range(1, len(prices)):
    total += prices[i]
''',
        why_flagged=(
            "Starting the range at 1 silently excludes `prices[0]` from the "
            "total. This kind of boundary error is easy to miss in review "
            "since the code runs without raising an exception."
        ),
        good_pattern='''
for i in range(len(prices)):
    total += prices[i]
# or more idiomatically:
total = sum(prices)
''',
        tags=["off-by-one", "range", "loop bounds"],
    ),
    CriteriaExample(
        category="Bug",
        subcategory="Shadowing a Built-in Name",
        severity="Low",
        bad_pattern='''
def process(list, dict, id):
    filtered = [x for x in list if x.id != id]
    return dict(filtered)
''',
        why_flagged=(
            "Naming parameters `list`, `dict`, or `id` shadows the built-in "
            "functions within that scope, which can cause confusing bugs "
            "later in the same function if the built-in is needed."
        ),
        good_pattern='''
def process(items, record_map, record_id):
    filtered = [x for x in items if x.id != record_id]
    return record_map(filtered)
''',
        tags=["shadowing", "builtins", "naming"],
    ),
    CriteriaExample(
        category="Bug",
        subcategory="Exception Handlers in Wrong Order",
        severity="Med",
        bad_pattern='''
try:
    risky_call()
except Exception:
    handle_generic()
except ValueError:
    handle_value_error()
''',
        why_flagged=(
            "Except clauses are checked in order, and `Exception` matches "
            "almost everything, so the more specific `ValueError` handler "
            "below it is unreachable dead code."
        ),
        good_pattern='''
try:
    risky_call()
except ValueError:
    handle_value_error()
except Exception:
    handle_generic()
''',
        tags=["exception order", "unreachable code"],
    ),
]

# ---------------------------------------------------------------------------
# 2. SECURITY — vulnerabilities and unsafe practices
# ---------------------------------------------------------------------------

SECURITY: List[CriteriaExample] = [
    CriteriaExample(
        category="Security",
        subcategory="Hardcoded Secret",
        severity="High",
        bad_pattern='''
API_KEY = "sk-I5GBu8CFgingkLXl5wFxdw"
client = OpenAI(api_key=API_KEY)
''',
        why_flagged=(
            "Secrets committed directly to source code are exposed to "
            "anyone with repo access and remain in git history even if "
            "removed later. They are also easy to leak accidentally via "
            "screenshots, logs, or forks."
        ),
        good_pattern='''
API_KEY = st.secrets.get("GENAI_API_KEY")
if not API_KEY:
    st.error("Missing API key")
    st.stop()
client = OpenAI(api_key=API_KEY)
''',
        tags=["secrets", "api key", "hardcoded credentials"],
    ),
    CriteriaExample(
        category="Security",
        subcategory="Disabled SSL Verification",
        severity="High",
        bad_pattern='''
requests.Session.request = lambda self, *a, **kw: old_request(
    self, *a, **{**kw, "verify": False}
)
client = httpx.Client(verify=False)
''',
        why_flagged=(
            "Disabling certificate verification globally removes protection "
            "against man-in-the-middle attacks for every request made "
            "through that client or session, not just the intended one. "
            "Even when done to work around a corporate proxy, it should be "
            "scoped as narrowly as possible and never applied process-wide."
        ),
        good_pattern='''
# Prefer trusting the corporate proxy's CA bundle explicitly,
# scoped to only the client that needs it:
client = httpx.Client(verify="/path/to/corporate-ca-bundle.pem")
''',
        tags=["ssl", "tls", "verify=False", "mitm"],
    ),
    CriteriaExample(
        category="Security",
        subcategory="SQL Injection via String Concatenation",
        severity="High",
        bad_pattern='''
query = "SELECT * FROM users WHERE username = '" + username + "'"
cursor.execute(query)
''',
        why_flagged=(
            "Building SQL by string concatenation allows an attacker to "
            "inject arbitrary SQL by controlling the `username` input, "
            "potentially bypassing authentication or exfiltrating data."
        ),
        good_pattern='''
cursor.execute(
    "SELECT * FROM users WHERE username = %s", (username,)
)
''',
        tags=["sql injection", "parameterized query"],
    ),
    CriteriaExample(
        category="Security",
        subcategory="Unsafe Deserialization",
        severity="High",
        bad_pattern='''
import pickle
data = pickle.loads(request.body)
''',
        why_flagged=(
            "`pickle.loads` on untrusted input can execute arbitrary code "
            "during deserialization, since pickle streams can encode object "
            "constructors and function calls, not just data."
        ),
        good_pattern='''
import json
data = json.loads(request.body)
''',
        tags=["deserialization", "pickle", "remote code execution"],
    ),
    CriteriaExample(
        category="Security",
        subcategory="eval/exec on User Input",
        severity="High",
        bad_pattern='''
result = eval(user_expression)
''',
        why_flagged=(
            "`eval` executes arbitrary Python, so any user-controlled "
            "string becomes a code execution vector. Even seemingly "
            "constrained inputs (e.g., 'just a math expression') can be "
            "crafted to break out via builtins."
        ),
        good_pattern='''
import ast
# For arithmetic-only expressions, use a restricted parser:
tree = ast.parse(user_expression, mode="eval")
# validate node types before evaluating, or use a library like `numexpr`
''',
        tags=["eval", "exec", "code injection"],
    ),
    CriteriaExample(
        category="Security",
        subcategory="Path Traversal",
        severity="High",
        bad_pattern='''
filename = request.args.get("file")
with open(f"/data/uploads/{filename}") as f:
    return f.read()
''',
        why_flagged=(
            "An attacker can pass `filename=../../etc/passwd` to escape the "
            "intended directory and read arbitrary files on the server, "
            "since the path is not validated or sandboxed."
        ),
        good_pattern='''
import os
safe_name = os.path.basename(filename)
full_path = os.path.join("/data/uploads", safe_name)
if not os.path.realpath(full_path).startswith("/data/uploads"):
    raise ValueError("Invalid path")
with open(full_path) as f:
    return f.read()
''',
        tags=["path traversal", "file access", "sandboxing"],
    ),
    CriteriaExample(
        category="Security",
        subcategory="Shell Injection via subprocess",
        severity="High",
        bad_pattern='''
os.system(f"convert {user_filename} output.png")
''',
        why_flagged=(
            "Passing user-controlled input into a shell command allows "
            "command injection, e.g. a filename like `a.png; rm -rf /` "
            "would execute an additional arbitrary command."
        ),
        good_pattern='''
import subprocess
subprocess.run(["convert", user_filename, "output.png"], check=True)
''',
        tags=["shell injection", "subprocess", "command injection"],
    ),
    CriteriaExample(
        category="Security",
        subcategory="Weak Hash for Passwords",
        severity="Med",
        bad_pattern='''
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()
''',
        why_flagged=(
            "MD5 (and SHA-1) are fast to compute, which makes them poorly "
            "suited for password storage: attackers can brute-force or "
            "rainbow-table crack them at high speed. Passwords need a slow, "
            "salted key-derivation function."
        ),
        good_pattern='''
import bcrypt
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
''',
        tags=["hashing", "passwords", "md5", "bcrypt"],
    ),
]

# ---------------------------------------------------------------------------
# 3. PERFORMANCE — complexity and efficiency hotspots
# ---------------------------------------------------------------------------

PERFORMANCE: List[CriteriaExample] = [
    CriteriaExample(
        category="Performance",
        subcategory="String Concatenation in Loop",
        severity="Med",
        bad_pattern='''
result = ""
for chunk in chunks:
    result += chunk
''',
        why_flagged=(
            "Strings are immutable, so `+=` creates a new string object on "
            "every iteration, making this O(n^2) for n chunks instead of "
            "O(n)."
        ),
        good_pattern='''
result = "".join(chunks)
''',
        tags=["string concatenation", "quadratic time", "join"],
    ),
    CriteriaExample(
        category="Performance",
        subcategory="Pandas iterrows Instead of Vectorization",
        severity="Med",
        bad_pattern='''
for index, row in df.iterrows():
    df.at[index, "total"] = row["price"] * row["qty"]
''',
        why_flagged=(
            "`iterrows()` is one of the slowest ways to operate on a "
            "DataFrame since it materializes each row as a Series. "
            "Vectorized operations use optimized C code under the hood and "
            "are orders of magnitude faster."
        ),
        good_pattern='''
df["total"] = df["price"] * df["qty"]
''',
        tags=["pandas", "vectorization", "iterrows"],
    ),
    CriteriaExample(
        category="Performance",
        subcategory="Membership Test on List Instead of Set",
        severity="Low",
        bad_pattern='''
allowed = ["admin", "editor", "viewer", ...]  # large list
if role in allowed:
    grant_access()
''',
        why_flagged=(
            "`in` on a list is O(n); on a set or dict it is O(1) on average. "
            "For repeated lookups against a large, static collection, using "
            "a set meaningfully reduces total work."
        ),
        good_pattern='''
allowed = {"admin", "editor", "viewer", ...}
if role in allowed:
    grant_access()
''',
        tags=["set", "list", "membership test", "complexity"],
    ),
    CriteriaExample(
        category="Performance",
        subcategory="N+1 Query Pattern",
        severity="High",
        bad_pattern='''
orders = Order.objects.all()
for order in orders:
    customer = Customer.objects.get(id=order.customer_id)
    print(customer.name)
''',
        why_flagged=(
            "This issues one query to fetch orders, then one additional "
            "query per order to fetch its customer — N+1 total queries. "
            "For large datasets this causes severe, easily overlooked "
            "database load."
        ),
        good_pattern='''
orders = Order.objects.select_related("customer").all()
for order in orders:
    print(order.customer.name)
''',
        tags=["n+1", "database", "orm", "select_related"],
    ),
    CriteriaExample(
        category="Performance",
        subcategory="Recompiling Regex Inside a Loop",
        severity="Low",
        bad_pattern='''
for line in lines:
    if re.match(r"^\\d{3}-\\d{4}$", line):
        matches.append(line)
''',
        why_flagged=(
            "`re.match` recompiles the pattern on every call unless it hits "
            "Python's internal regex cache. For hot loops with complex "
            "patterns, explicit compilation avoids repeated parsing "
            "overhead and makes intent clearer."
        ),
        good_pattern='''
pattern = re.compile(r"^\\d{3}-\\d{4}$")
for line in lines:
    if pattern.match(line):
        matches.append(line)
''',
        tags=["regex", "compilation", "loop"],
    ),
    CriteriaExample(
        category="Performance",
        subcategory="Unnecessary Deep Copy",
        severity="Low",
        bad_pattern='''
import copy
for record in records:
    snapshot = copy.deepcopy(record)
    read_only_process(snapshot)
''',
        why_flagged=(
            "Deep-copying is expensive and only necessary when the copy "
            "will be mutated. If the function only reads the data, copying "
            "adds overhead with no benefit."
        ),
        good_pattern='''
for record in records:
    read_only_process(record)
''',
        tags=["deepcopy", "unnecessary copy"],
    ),
    CriteriaExample(
        category="Performance",
        subcategory="Repeated Recomputation Inside Loop",
        severity="Med",
        bad_pattern='''
for item in items:
    threshold = compute_expensive_threshold(config)
    if item.value > threshold:
        flag(item)
''',
        why_flagged=(
            "`compute_expensive_threshold(config)` does not depend on "
            "`item`, yet it is recomputed on every iteration instead of "
            "once before the loop, wasting repeated work."
        ),
        good_pattern='''
threshold = compute_expensive_threshold(config)
for item in items:
    if item.value > threshold:
        flag(item)
''',
        tags=["loop invariant", "recomputation"],
    ),
]

# ---------------------------------------------------------------------------
# 4. STYLE — readability, maintainability, and Pythonic conventions
# ---------------------------------------------------------------------------

STYLE: List[CriteriaExample] = [
    CriteriaExample(
        category="Style",
        subcategory="Deep Nesting Instead of Early Return",
        severity="Low",
        bad_pattern='''
def process(user):
    if user is not None:
        if user.is_active:
            if user.has_permission:
                return do_work(user)
            else:
                return None
        else:
            return None
    else:
        return None
''',
        why_flagged=(
            "Deeply nested conditionals make the happy path hard to find "
            "and increase cyclomatic complexity. Guard clauses that return "
            "early flatten the structure and make the main logic obvious."
        ),
        good_pattern='''
def process(user):
    if user is None or not user.is_active or not user.has_permission:
        return None
    return do_work(user)
''',
        tags=["nesting", "guard clause", "early return"],
    ),
    CriteriaExample(
        category="Style",
        subcategory="Missing Type Hints on Public Function",
        severity="Low",
        bad_pattern='''
def calculate_total(items, tax_rate):
    return sum(i.price for i in items) * (1 + tax_rate)
''',
        why_flagged=(
            "Public functions without type hints force callers and tools "
            "(IDEs, linters, mypy) to infer types manually, increasing the "
            "chance of misuse and making refactors riskier."
        ),
        good_pattern='''
def calculate_total(items: list[LineItem], tax_rate: float) -> float:
    return sum(i.price for i in items) * (1 + tax_rate)
''',
        tags=["type hints", "typing", "public api"],
    ),
    CriteriaExample(
        category="Style",
        subcategory="Magic Numbers",
        severity="Low",
        bad_pattern='''
if user.age >= 18 and account.balance > 500:
    approve_loan()
''',
        why_flagged=(
            "Unnamed numeric literals hide their meaning and make future "
            "changes error-prone, since a reader has to guess whether "
            "'18' and '500' are business rules, coincidences, or "
            "placeholders."
        ),
        good_pattern='''
MIN_ADULT_AGE = 18
MIN_BALANCE_FOR_LOAN = 500

if user.age >= MIN_ADULT_AGE and account.balance > MIN_BALANCE_FOR_LOAN:
    approve_loan()
''',
        tags=["magic numbers", "constants", "readability"],
    ),
    CriteriaExample(
        category="Style",
        subcategory="Function Doing Too Much",
        severity="Med",
        bad_pattern='''
def handle_order(order):
    # validates, charges payment, updates inventory,
    # sends email, and logs analytics all in one function
    ...
''',
        why_flagged=(
            "A single function responsible for validation, payment, "
            "inventory, notification, and analytics is hard to test, reuse, "
            "or reason about in isolation, and a bug in one concern risks "
            "breaking unrelated ones."
        ),
        good_pattern='''
def handle_order(order):
    validate_order(order)
    charge_payment(order)
    update_inventory(order)
    send_confirmation_email(order)
    log_order_event(order)
''',
        tags=["single responsibility", "function length", "refactor"],
    ),
    CriteriaExample(
        category="Style",
        subcategory="Using print Instead of Logging",
        severity="Low",
        bad_pattern='''
print(f"Failed to process {record_id}: {error}")
''',
        why_flagged=(
            "`print` statements can't be filtered by severity, redirected "
            "to log aggregation systems, or disabled in production without "
            "code changes. Logging gives structured, configurable output."
        ),
        good_pattern='''
import logging
logger = logging.getLogger(__name__)
logger.error("Failed to process %s: %s", record_id, error)
''',
        tags=["logging", "print", "observability"],
    ),
    CriteriaExample(
        category="Style",
        subcategory="Wildcard Import",
        severity="Low",
        bad_pattern='''
from utils import *
''',
        why_flagged=(
            "Wildcard imports pollute the namespace with unknown names, "
            "make it unclear where a given function came from, and can "
            "silently shadow existing names."
        ),
        good_pattern='''
from utils import parse_date, format_currency
''',
        tags=["wildcard import", "namespace", "imports"],
    ),
    CriteriaExample(
        category="Style",
        subcategory="Unclear Single-Letter Naming Outside Tight Scope",
        severity="Low",
        bad_pattern='''
def calc(a, b, c):
    return a * b + c if c > 0 else a * b
''',
        why_flagged=(
            "Single-letter names are fine as loop counters in a two-line "
            "scope, but in a function signature they force every caller and "
            "reader to guess what the parameters represent."
        ),
        good_pattern='''
def calculate_price(base_price: float, quantity: int, discount: float) -> float:
    return base_price * quantity + discount if discount > 0 else base_price * quantity
''',
        tags=["naming", "readability"],
    ),
]

# ---------------------------------------------------------------------------
# Combined taxonomy + helper to build LangChain Documents
# ---------------------------------------------------------------------------

REVIEW_CRITERIA: List[CriteriaExample] = BUGS + SECURITY + PERFORMANCE + STYLE


def build_criteria_documents(criteria: List[CriteriaExample] = REVIEW_CRITERIA) -> List[Document]:
    """
    Convert the criteria taxonomy into LangChain Documents suitable for
    embedding into a Chroma vectorstore. The page_content is what gets
    embedded and matched against code chunks at review time; the good_pattern,
    severity, and category are kept in metadata for use in the prompt.
    """
    documents = []
    for c in criteria:
        page_content = (
            f"Category: {c.category} - {c.subcategory}\n"
            f"Problematic pattern:\n{c.bad_pattern.strip()}\n\n"
            f"Why this is flagged: {c.why_flagged}"
        )
        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "category": c.category,
                    "subcategory": c.subcategory,
                    "severity": c.severity,
                    "good_pattern": c.good_pattern.strip(),
                    "tags": ", ".join(c.tags),
                },
            )
        )
    return documents


if __name__ == "__main__":
    # Quick sanity check when run directly: print counts per category.
    from collections import Counter
    counts = Counter(c.category for c in REVIEW_CRITERIA)
    print(f"Total criteria examples: {len(REVIEW_CRITERIA)}")
    for category, count in counts.items():
        print(f"  {category}: {count}")
