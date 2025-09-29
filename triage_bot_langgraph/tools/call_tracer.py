import sys
import inspect
import pytest
import linecache
import weakref

# Optional tensor library detection
try:
    import torch
except Exception:
    torch = None

_prev_locals = {}

def _is_tensor(obj):
    if torch is not None and isinstance(obj, torch.Tensor):
        return True
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "__array__")

def _render_value(v, max_len=120):
    try:
        if _is_tensor(v):
            shape = getattr(v, "shape", None)
            return f"<Tensor shape={tuple(shape)} if shape is not None else '?' >"
        if isinstance(v, (int, float, bool, type(None), str)):
            s = repr(v)
        elif isinstance(v, (list, tuple, set, frozenset)):
            s = f"<{type(v).__name__} len={len(v)}>"
        elif isinstance(v, dict):
            s = f"<dict len={len(v)}>"
        else:
            s = f"<{type(v).__name__}>"
        if len(s) > max_len:
            s = s[: max_len - 3] + "..."
        return s
    except Exception as e:
        return f"<unrepr {type(v).__name__}: {e}>"

def _diff_locals(frame):
    fid = id(frame)
    current = frame.f_locals.copy()
    old = _prev_locals.get(fid, {})
    changes = {}
    for k, v in current.items():
        if k.startswith("__") and k.endswith("__"):
            continue
        if k not in old or old[k] is not v or old[k] != v:
            changes[k] = _render_value(v)
    removed = [k for k in old.keys() if k not in current]
    _prev_locals[fid] = current
    return changes, removed

def trace_calls2(frame, event, arg):
    code_obj = frame.f_code
    func_name = code_obj.co_name
    filename = code_obj.co_filename
    lineno = frame.f_lineno

    if event == "call":
        # Initialize locals snapshot
        _prev_locals[id(frame)] = frame.f_locals.copy()
        print(f"CALL2: {func_name}() at {filename}:{lineno}")
        if frame.f_locals:
            rendered = {k: _render_value(v) for k, v in frame.f_locals.items() if not (k.startswith('__') and k.endswith('__'))}
            if rendered:
                print(f"\tArgs: {rendered}")
    elif event == "line":
        changes, removed = _diff_locals(frame)
        if changes or removed:
            msg_parts = []
            if changes:
                msg_parts.append("updated=" + ", ".join(f"{k}={v}" for k, v in changes.items()))
            if removed:
                msg_parts.append("removed=" + ", ".join(removed))
            print(f"LINE2: {filename}:{lineno} in {func_name} -> {linecache.getline(filename, lineno).rstrip()}")

            print(f"\tLocals: { '; '.join(msg_parts) }")
    elif event == "return":
        fid = id(frame)
        if fid in _prev_locals:
            del _prev_locals[fid]
        print(f"RETURN2: {func_name}() -> {_render_value(arg)}")
    elif event == "exception":
        etype, evalue, _ = arg
        print(f"EXCEPTION2: {func_name} {etype.__name__}: {evalue}")
    return trace_calls2

# Redefine trace_calls to also print the actual source line executed
def trace_calls(frame, event, arg):
    code_obj = frame.f_code
    func_name = code_obj.co_name
    filename = code_obj.co_filename
    lineno = frame.f_lineno

    if event == "call":
        
        src = linecache.getline(filename, lineno).rstrip()

        call_type = "function"
        owner = None
        try:
            if 'self' in frame.f_locals:
                self_obj = frame.f_locals['self']
                cls = type(self_obj)
                owner = cls.__name__
                if func_name == "__init__":
                    call_type = "constructor"
                else:
                    raw = getattr(cls, func_name, None)
                    if isinstance(raw, classmethod):
                        call_type = "classmethod"
                    elif isinstance(raw, staticmethod):
                        call_type = "staticmethod"
                    else:
                        call_type = "method"
            elif 'cls' in frame.f_locals and inspect.isclass(frame.f_locals['cls']):
                cls = frame.f_locals['cls']
                owner = cls.__name__
                raw = getattr(cls, func_name, None)
                if isinstance(raw, classmethod):
                    call_type = "classmethod"
                elif isinstance(raw, staticmethod):
                    call_type = "staticmethod"
                else:
                    if func_name == "__init__":
                        call_type = "constructor"
                    else:
                        call_type = "method"
            elif func_name == "__init__":
                call_type = "constructor"
        except Exception:
            pass

        try:
            arginfo = inspect.getargvalues(frame)
            arg_parts = []
            for name in arginfo.args:
                if name in arginfo.locals:
                    arg_parts.append(f"{name}={_render_value(arginfo.locals[name])}")
            if arginfo.varargs:
                varargs_val = arginfo.locals.get(arginfo.varargs, ())
                arg_parts.append(f"*{arginfo.varargs}={_render_value(varargs_val)}")
            if arginfo.keywords:
                kwargs_val = arginfo.locals.get(arginfo.keywords, {})
                arg_parts.append(f"**{arginfo.keywords}={_render_value(kwargs_val)}")
            arg_str = ", ".join(arg_parts)
        except Exception:
            arg_str = "?"

        owner_part = f"{owner}." if owner else ""
        print(f"CALL: {call_type} {owner_part}{func_name}({arg_str}) at {filename}:{lineno}")
        print(f"\tSource: {src}")
        user_locals = {
            k: _render_value(v)
            for k, v in frame.f_locals.items()
            if not (k.startswith('__') and k.endswith('__'))
        }
        if user_locals:
            print(f"\tLocals: {user_locals}")
    elif event == "line":
        src = linecache.getline(filename, lineno).rstrip()
        print(f"LINE: {filename}:{lineno} in {func_name} -> {src}")
    elif event == "return":
        print(f"RETURN: {func_name}() -> {arg}")
        print(f"RETURN: {func_name}()")
    elif event == "exception":
        print(f"EXCEPTION in {func_name}: {arg[0].__name__}: {arg[1]}")
    return trace_calls


################################
# Pytest hook 
################################

# To have pytest invoke pytest_runtest_protocol you do NOT call it directly.
# Make sure this module is loaded as a pytest plugin, then just run pytest.
#
# Option A (recommended): In some conftest.py (e.g. project root) add:
#     pytest_plugins = ["ai_for_validation.triage_bot_langgraph.common"]
#
# Option B: Load explicitly via command line:
#     pytest -p ai_for_validation.triage_bot_langgraph.common -s
#
# Option C: Programmatic (from another script):
#     import pytest
#     pytest.main(["-p", "ai_for_validation.triage_bot_langgraph.common", "-s", "tests/"])
#
# Use -s so the print from the hook is shown.
#
# Example:
#     pytest -p ai_for_validation.triage_bot_langgraph.tools.call_tracer -s tests/test_example.py::test_something
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    print(f"[triage] START {item.nodeid}")
    sys.settrace(trace_calls)
    try:
        yield
    finally:
        sys.settrace(None)
        print(f"[triage] END {item.nodeid}")
   


# # Enable tracing
# sys.settrace(trace_calls)

# # Example functions to trace
# def calculate_sum(a, b):
#     result = a + b
#     return result

# def process_data(data):
#     total = 0
#     for item in data:
#         total += calculate_sum(item, 1)
#     return total

# # This will be traced
# data = [1, 2, 3, 4, 5]
# result = process_data(data)
# print(f"Final result: {result}")

# # Disable tracing
# sys.settrace(None)


# def trace_calls(frame, event, arg):
#     """Custom trace function that intercepts all function calls"""
#     if event == 'call':
#         # Get information about the called function
#         code = frame.f_code
#         func_name = code.co_name
#         filename = code.co_filename
#         line_no = frame.f_lineno
        
#         # Get source code if available
#         try:
#             source_lines, start_line = inspect.getsourcelines(frame)
#             current_line = line_no - start_line
#             if 0 <= current_line < len(source_lines):
#                 source_code = source_lines[current_line].strip()
#             else:
#                 source_code = "N/A"
#         except:
#             source_code = "N/A"
        
#         print(f"CALL: {func_name}() at {filename}:{line_no}")
#         print(f"\tSource: {source_code}")
#         print(f"\tLocals: {list(frame.f_locals.keys())}")
    
#     elif event == 'line':
#         # Called for each line execution
#         print(f"LINE: {frame.f_lineno} in {frame.f_code.co_name}")
    
#     elif event == 'return':
#         print(f"RETURN: {frame.f_code.co_name}() -> {arg}")
    
#     elif event == 'exception':
#         print(f"EXCEPTION: {arg[0].__name__}: {arg[1]}")
    
#     return trace_calls  # Return itself to continue tracing
