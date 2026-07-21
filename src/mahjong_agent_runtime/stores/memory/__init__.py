"""In-memory backend implementation modules.

Concrete mixins are deliberately imported by ``store.py`` only after its
compatibility domain helpers have been defined. Eager imports here would
re-enter that partially initialized module and create a cycle.
"""
