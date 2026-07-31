"""
Backward compatibility builder wrapper for InkFlow-AI graph.
"""

from src.graph.main_graph import build_main_graph as build_graph

__all__ = ["build_graph"]