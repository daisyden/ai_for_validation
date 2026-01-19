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
                args_info.append(f"{key}={repr(value)}")
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
        if _call_depth <= 5:
            src = linecache.getline(filename, lineno).strip()
            filename = filename.split("site-packages/")[-1]
            if any(word in filename for word in ["torch\/test\/", "torch\/aten\/", "torch\/torch\/"] ):
                print(f"[triage]{indent}CALL: {func_name}() at {filename}:{lineno} -> {src}")
                print(f"[triage]{indent}  Arguments: {format_arguments(frame)}")
                print(f"[triage]{indent}  Locals: {list(frame.f_locals.keys())}")
        _call_depth += 1
    elif event == "line":
        if _call_depth <= 5:
            src = linecache.getline(filename, lineno).rstrip()
            filename = filename.split("site-packages/")[-1]
            if any(word in filename for word in ["torch\/test\/", "torch\/aten\/", "torch\/torch\/"] ):
                print(f"[triage]{indent}LINE: {filename}:{lineno} in {func_name} -> {src}")
                print(f"[triage]{indent}  Arguments: {format_arguments(frame)}")
                print(f"[triage]{indent}  Locals: {list(frame.f_locals.keys())}")
    elif event == "return":
        _call_depth = max(0, _call_depth - 1)
        if _call_depth <= 5:
            indent = "  " * _call_depth
            if any(word in filename for word in ["torch\/test\/", "torch\/aten\/", "torch\/torch\/"] ):
                print(f"[triage]{indent}RETURN: {func_name}() -> {arg}")
    elif event == "exception":
        print(f"[triage]{indent}EXCEPTION in {func_name}: {arg[0].__name__}: {arg[1]}")
    return trace_calls


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    print(f"[triage] START {item.nodeid}")
    # Get the test function body
    test_func = item.obj
    if hasattr(test_func, '__code__'):
        code = test_func.__code__
        filename = code.co_filename
        start_line = code.co_firstlineno
        # Read the source lines for the function
        lines = []
        try:
            source_lines = inspect.getsourcelines(test_func)[0]
            # Skip the function definition line (def test_...():)
            for line in source_lines[1:]:
                lines.append(line.rstrip())
        except:
            pass
        if lines:
            print(f"[triage] Test body:\n" + "\n".join(lines))
    
    print("\n[triage] Call trace:\n")
    sys.settrace(trace_calls)
    try:
        yield
    finally:
        sys.settrace(None)
        print(f"[triage] END {item.nodeid}")
