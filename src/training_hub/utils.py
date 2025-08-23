from typing import get_origin, get_args

def format_type_name(tp):
    # Handle None
    if tp is type(None):
        return 'None'
    
    # Handle basic types
    if hasattr(tp, '__name__'):
        return tp.__name__
    
    # Handle typing generics
    origin = get_origin(tp)
    args = get_args(tp)
    
    if origin is not None:
        origin_name = getattr(origin, '__name__', str(origin))
        if args:
            arg_names = [format_type_name(arg) for arg in args]
            return f"{origin_name}[{', '.join(arg_names)}]"
        return origin_name
    
    # Fallback: clean up the string representation
    type_str = str(tp)
    if type_str.startswith("<class '") and type_str.endswith("'>"):
        return type_str[8:-2]
    
    return type_str
