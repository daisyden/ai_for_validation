import sys
import pytest
import linecache
import inspect

def format_arguments(frame):
    """Format function arguments or state variables for tracing output.

    For tensors: prints shape and dtype
    For iterables: prints type and length
    For others: prints repr
    """
    args_info = []
    for key, value in frame.f_locals.items():
        if key == 'self':
            continue
        try:
            # Check if it's a tensor (PyTorch or TensorFlow)
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                args_info.append(f"{key}=Tensor(shape={value.shape}, dtype={value.dtype})")
            # Check if it's a list or iterable (but not string)
            elif isinstance(value, (list, tuple, set)):
                dtype = type(value).__name__
                args_info.append(f"{key}={dtype}(len={len(value)})")
            # Handle other iterables with __len__")
            elif hasattr(value, '__iter__') and hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                dtype = type(value).__name__
                args_info.append(f"{key}={dtype}(len={len(value)})")
            else:
                # Limit the length of repr to avoid printing huge values
                value_repr = repr(value)
                if len(value_repr) > 64:
                    value_repr = value_repr[:61] + "..."
                args_info.append(f"{key}={value_repr}")
        except:
            args_info.append(f"{key}=<error>")
    return ', '.join(args_info)



# Redefine trace_calls to also print the actual source line executed
_call_depth = 0

def trace_calls(frame, event, arg):
    global _call_depth

    code_obj = frame.f_code
    func_name = code_obj.co_name
    filename = code_obj.co_filename
    lineno = frame.f_lineno

    # Create indentation based on call depth
    indent = "  " * _call_depth

    if event == "call":
        if True:
            src = linecache.getline(filename, lineno).strip()
            #filename = filename.split("site-packages/")[-1]
            if any(word in filename for word in ["/test/"] ):
                    print(f"[triage]{indent}CALL: {func_name}() at {filename}:{lineno} -> {src}")
                    print(f"[triage]{indent}  Arguments: {format_arguments(frame)}")
                    print(f"[triage]{indent}  Locals: {list(frame.f_locals.keys())}")
        _call_depth += 1
    elif event == "line":
       if _call_depth == 1:
           src = linecache.getline(filename, lineno).rstrip()
           #filename = filename.split("site-packages/")[-1]
           if any(word in filename for word in ["/test/"] ):
                   print(f"[triage]{indent}LINE: {filename}:{lineno} in {func_name} -> {src}")
                   print(f"[triage]{indent}  Arguments: {format_arguments(frame)}")
                   print(f"[triage]{indent}  Locals: {list(frame.f_locals.keys())}")
    elif event == "return":
        _call_depth = max(0, _call_depth - 1)
        if True:
            indent = "  " * _call_depth
            if any(word in filename for word in ["/test/"] ):
                    #print(f"[triage]{indent}RETURN: {func_name}() -> {arg}")
                    print(f"[triage]{indent}RETURN: {func_name}()")
    #elif event == "exception":
    #    print(f"[triage]{indent}EXCEPTION in {func_name}: {arg[0].__name__}: {arg[1]}")
    return trace_calls


def pytest_runtest_setup(item):
    sys.settrace(trace_calls)

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    print(f"[triage] START {item.nodeid}")
    # Get the test function body
    test_func = item.obj
    module = inspect.getmodule(test_func)
    module_file = inspect.getfile(module)
    # Get the test file name
    test_file = module_file
    print(f"[triage] Test file: {test_file}")

    import ast

    def extract_imports_from_file(filepath):
        with open(filepath, 'r') as file:
            tree = ast.parse(file.read())

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module:
                        imports.append(f"{module}.{alias.name}")
                    else:
                        imports.append(alias.name)

        return imports

    # Usage
    imports = extract_imports_from_file(test_file)
    print(f"[triage] Imports: {imports}")

    # Read the source lines for the function
    if hasattr(test_func, '__code__'):

        lines = []
        try:
            source_lines = inspect.getsourcelines(test_func)[0]

            for line in source_lines:
                lines.append(line.rstrip())
        except:
            pass

        if lines:
            print(f"[triage] Test body:\n" + "\n".join(lines))

    print("\n[triage] Call trace:\n")
    #sys.settrace(trace_calls)
    try:
        yield
    finally:
        sys.settrace(None)
        print(f"[triage] END {item.nodeid}")
