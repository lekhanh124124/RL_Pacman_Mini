# Custom CSS cho phong cách Glassmorphism và Dark Mode hiện đại của Pacman Mini RL Dashboard

CUSTOM_CSS = """
<style>
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1f2e;
        border-radius: 10px;
        color: #94a3b8;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #2e374f;
        color: #38bdf8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
    }
    .metric-card {
        background-color: #1a1f2e;
        border: 1px solid #2e374f;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: bold;
        color: #38bdf8;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Maze Board CSS Styles for Stable Grid Alignment */
    .maze-table {
        border-collapse: separate;
        border-spacing: 4px;
        margin: 0 auto;
    }
    .maze-cell {
        width: 45px;
        height: 45px;
        text-align: center;
        vertical-align: middle;
        font-size: 24px;
        border-radius: 8px;
        background-color: #1e293b;
        border: 1px solid #334155;
        transition: all 0.15s ease-in-out;
    }
    .cell-wall {
        background-color: #0f172a;
        border: 2px solid #3b82f6;
        box-shadow: inset 0 0 10px rgba(59, 130, 246, 0.4);
    }
    .cell-danger {
        background-color: rgba(239, 68, 68, 0.15);
        border: 1.5px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.2);
    }
    .cell-pacman {
        background-color: rgba(234, 179, 8, 0.25);
        border: 2px solid rgba(234, 179, 8, 0.8);
        box-shadow: 0 0 15px rgba(234, 179, 8, 0.5);
    }
    .cell-ghost {
        background-color: rgba(239, 68, 68, 0.35);
        border: 2px solid rgba(239, 68, 68, 0.9);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.7);
    }
</style>
"""
