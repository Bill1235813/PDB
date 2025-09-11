bug_type_examples = [
    "Adding ONE line",
    "Deleting ONE line",
    "Modifying ONE line, not adding or deleting. Some bug examples (NO need to follow these examples exactly):\n[[bug_description]]",
]

hard_bug_examples = {
    "Mutable Default Arguments": """A list, dictionary, or other mutable object used as a default argument is created only once when the function is defined. Subsequent calls to the function without that argument will modify the same object, leading to unexpected shared state across calls.
Bug Example: def add_to_list(item, my_list=[]): ...""",

    "Late Binding in Closures": """Variables in loops are bound by reference, not value. A lambda or function defined in a loop will use the final value of the loop variable when it's eventually called, not the value it had when the function was defined.
Bug Example: multipliers = [lambda x: i * x for i in range(5)]  # All lambdas will use i=4.""",

    "List Multiplication Surprise": """Using `[[]] * N` to create a list of lists results in a list containing N references to the *very same* inner list. Modifying one inner list (e.g., matrix[0].append(1)) will appear to modify all of them.
Bug Example: matrix = [[]] * 4""",

    "Modifying a List While Iterating": """Removing or adding elements to a list while iterating over it directly disrupts the iterator's internal index. This causes the loop to skip over elements immediately following a removed item.
Bug Example: for item in my_list: my_list.remove(item)""",

    "The 'is' vs. '==' Trap": """The `is` operator checks for object identity, while `==` checks for value equality. Python's CPython implementation caches small integers (-5 to 256), so `is` works for them but fails for larger integers that have the same value but are different objects in memory.
Bug Example: a = 257; b = 257; if a is b: ... # This is False.""",

    "In-Place `sort()` Method Returns `None`": """The `list.sort()` method sorts a list in-place and returns `None`. A common mistake is to assign the result to a new variable, which will then be `None`, leading to TypeErrors later.
Bug Example: sorted_list = my_list.sort()""",

    "Accidental Tuple Creation": """A stray trailing comma after a variable assignment will quietly convert it into a single-element tuple. Subsequent code expecting a string, integer, etc., will fail.
Bug Example: my_var = "some_value",""",

    "The `or` Operator for Defaulting": """Using `variable = user_input or "default"` fails when a valid input is a "falsy" value (like an empty string `""`, the number `0`, or `False`), as it will incorrectly trigger the default case.
Bug Example: name = "" or "default_user" # name becomes "default_user".""",

    "Out-of-Bounds Slicing Fails Silently": """Unlike single-index access, slicing a list with an out-of-bounds range does not raise an `IndexError`. It silently returns an empty list, which can hide bugs in index calculation logic.
Bug Example: sub_list = [1, 2, 3][10:20] # sub_list is now []""",

    "`datetime.now()` as a Default Argument": """Similar to mutable defaults, using `datetime.now()` as a default argument captures the timestamp only once—at function definition time—not each time the function is called.
Bug Example: def log(msg, ts=datetime.now()): ...""",

    "String `strip()` Misunderstanding": """`'string'.strip('chars')` does not remove the literal substring 'chars'. It removes any character present in the set {'c', 'h', 'a', 'r', 's'} from both ends of the string.
Bug Example: "example.com".strip(".com") # Results in "xampl" """,

    "Overwriting Built-in Names": """Assigning a value to a variable name that shadows a built-in function (e.g., `list`, `sum`, `dict`) will cause `TypeError`s when the code later tries to call the original built-in function using that name.
Bug Example: list = [1, 2, 3] # Later, list((4, 5)) fails.""",

    "Floating-Point Inaccuracy": """Standard binary floating-point numbers cannot represent some decimal fractions exactly. Direct comparison of floating-point numbers can lead to unexpected `False` results due to precision errors.
Bug Example: if 0.1 + 0.2 == 0.3: ... # This is False.""",

    "Greedy Regex Matching": """By default, regex quantifiers like `*` and `+` are greedy, meaning they match the longest possible string. This can lead to incorrect results when parsing nested structures.
Bug Example: re.match('<.*>', '<h1>title</h1>') # Matches the whole string.""",

    "Exhausting an Iterator": """Iterators (like file objects, `zip` objects, or generators) can only be consumed once. Attempting to iterate over them a second time will yield no items, as the iterator is already exhausted.
Bug Example: with open('f.txt') as f: print(list(f)); print(list(f)) # Second list is empty.""",

    "Chained Boolean Comparison Logic": """An expression like `if x in my_list == True:` is parsed by Python as `(x in my_list) and (my_list == True)`. The second part is almost always `False`, making the entire expression fail unexpectedly.
Bug Example: if "a" in ["a", "b"] == True: ... # This is False.""",

    "Single-Element Tuple Formatting with `%`": """The old `%` style string formatting can raise a `TypeError` if you provide a single argument that is itself a tuple, as it tries to unpack it. It must be wrapped in another tuple.
Bug Example: print("Value: %s" % (('x', 'y'))) # Fails without a trailing comma.""",

    "Pandas Chained Indexing Assignment": """Assigning a value to a pandas DataFrame using chained indexing (e.g., `df[...][...] = value`) often operates on a temporary copy of the data, not the original DataFrame. The assignment may fail silently or raise a `SettingWithCopyWarning`.
Bug Example: df[df['col1'] > 10]['col2'] = 99""",

    "Implicit Boolean Conversion of Collections": """Checking a collection in a boolean context (e.g., `if not my_list:`) evaluates to `True` for both an empty collection (`[]`) and `None`. This can hide the important logical distinction between 'no data provided' and 'an empty set of data'.
Bug Example: if not records: ... # Runs for both records=None and records=[]""",

    "`any()` on an Empty Iterable": """The function `any()` returns `False` for an empty iterable. While logically correct (there isn't 'any' true item), it can be a bug if the code doesn't distinguish between the case where 'all items are false' and the case where 'there are no items to check'.
Bug Example: if any(filtered_list): ... # Does not distinguish why it's False."""
}