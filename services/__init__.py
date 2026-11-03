"""
LemGendary Dataset Compiler — Services Layer
============================================
Encapsulates all domain workflows into dedicated single-responsibility services
(a function for a single operation, a class for a family of operations),
allowing in-process programmatic invocation from the unified CLI, tests,
and external applications without relying on subprocess shelling.

Import services from their modules (``from services.compiler_service import
run_compile``). The package itself imports nothing: long-running API jobs run
a service module as its own process (``python -m services.<module>``, see
:mod:`services.invocation`), and an eager package import would load that
module a second time before it runs as ``__main__``.
"""
