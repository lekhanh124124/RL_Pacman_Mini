import streamlit as st
import os

def get_maze_html(env, show_danger=True, policy_arrows=None):
    """
    Tạo chuỗi HTML biểu diễn mê cung dưới dạng bảng với kích thước ô cố định,
    ngăn ngừa tình trạng biến dạng khi Pacman/Ghost di chuyển và cải thiện thẩm mỹ.
    """
    html = "<table class='maze-table'>"
    for r in range(env.width):
        html += "<tr>"
        for c in range(env.height):
            pos = (r, c)
            cell_class = "maze-cell"
            cell_content = ""
            
            # 1. Xác định vật cản tường
            if env.grid[r][c] == 1:
                cell_class += " cell-wall"
                cell_content = "🧱"
            # 2. Vị trí Pacman
            elif not policy_arrows and pos == env.pacman_pos:
                cell_class += " cell-pacman"
                cell_content = "😮"
            # 3. Vị trí Ghost
            elif not policy_arrows and pos == env.ghost_pos:
                cell_class += " cell-ghost"
                cell_content = "👻"
            # 4. Các đồng xu
            elif not policy_arrows and pos in env.coins:
                cell_content = "🪙"
            # 5. Nếu hiển thị chính sách (Policy Map)
            elif policy_arrows and pos in policy_arrows:
                arrow = policy_arrows[pos]
                if arrow == "⚫":
                    cell_content = "<span style='color: #4b5563;'>⚫</span>"
                else:
                    cell_content = f"<span style='color: #38bdf8; font-weight: bold;'>{arrow}</span>"
            # 6. Vùng nguy hiểm gần Ghost
            else:
                dist_to_ghost = abs(r - env.ghost_pos[0]) + abs(c - env.ghost_pos[1])
                is_danger = show_danger and dist_to_ghost <= 1 and pos != env.ghost_pos
                
                if is_danger:
                    cell_class += " cell-danger"
                    cell_content = "⚠️"
                else:
                    # Ô trống di chuyển thông thường, hiển thị dấu chấm nhỏ dạng Pacman Pellet
                    cell_content = "<span style='color: #4b5563; font-size: 14px;'>●</span>"
                    
            html += f"<td class='{cell_class}'>{cell_content}</td>"
        html += "</tr>"
    html += "</table>"
    return html

def render_stats_table(size_data):
    """
    Tạo bảng số liệu thống kê dưới dạng HTML đẹp mắt.
    """
    table_rows = ""
    for agent_name, val in size_data.items():
        table_rows += "<tr>"
        table_rows += f"<td style='padding: 12px; font-weight: bold; border-bottom: 1px solid #2e374f;'>{agent_name}</td>"
        table_rows += f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #2e374f; color: #2ca02c;'>{val['success_rate']}</td>"
        table_rows += f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #2e374f; color: #d62728;'>{val['death_rate']}</td>"
        table_rows += f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #2e374f;'>{val['reward']}</td>"
        table_rows += f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #2e374f;'>{val['steps']}</td>"
        table_rows += f"<td style='padding: 12px; text-align: center; border-bottom: 1px solid #2e374f; color: #38bdf8;'>{val.get('steps_on_win', 'n/a')}</td>"
        table_rows += "</tr>"
        
    table_html = "<table style='width: 100%; border-collapse: collapse; background-color: #1a1f2e; border: 1px solid #2e374f; border-radius: 8px;'>"
    table_html += "<thead><tr style='background-color: #0f172a; color: white;'>"
    table_html += "<th style='padding: 12px; text-align: left;'>Thuật toán / Agent</th>"
    table_html += "<th style='padding: 12px; text-align: center;'>Tỷ lệ Thắng (Ăn hết xu)</th>"
    table_html += "<th style='padding: 12px; text-align: center;'>Tỷ lệ bị Ghost bắt (Thua)</th>"
    table_html += "<th style='padding: 12px; text-align: center;'>Điểm Reward tích lũy</th>"
    table_html += "<th style='padding: 12px; text-align: center;'>Số bước đi trung bình</th>"
    table_html += "<th style='padding: 12px; text-align: center;'>Số bước khi THẮNG</th>"
    table_html += "</tr></thead>"
    table_html += f"<tbody>{table_rows}</tbody>"
    table_html += "</table>"
    return table_html

def render_evaluation_plots(size_key):
    """
    Hiển thị 6 biểu đồ so sánh đối chứng theo bố cục lưới 3 hàng x 2 cột.
    """
    st.subheader(f"📈 Đồ thị so sánh đối chứng - Bản đồ {size_key}")
    
    # Hàng 1: Learning Curves và Success/Death Comparison
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        fig1_path = f"src/reports/figures/learning_curves_{size_key}.png"
        if os.path.exists(fig1_path):
            st.image(fig1_path, caption=f"Biểu đồ 1: Đường cong học tập trung bình trượt qua 10 Seeds - Bản đồ {size_key}", use_container_width=True)
    with col_fig2:
        fig2_path = f"src/reports/figures/success_death_comparison_{size_key}.png"
        if os.path.exists(fig2_path):
            st.image(fig2_path, caption=f"Biểu đồ 2: So sánh Tỷ lệ thắng & Tỷ lệ thua bị ma bắt - Bản đồ {size_key}", use_container_width=True)
            
    # Hàng 2: Steps Comparison và Wall Hits Comparison
    st.markdown("---")
    col_fig3, col_fig4 = st.columns(2)
    with col_fig3:
        fig3_path = f"src/reports/figures/steps_comparison_{size_key}.png"
        if os.path.exists(fig3_path):
            st.image(fig3_path, caption=f"Biểu đồ 3: So sánh Số bước đi trung bình mỗi Episode - Bản đồ {size_key}", use_container_width=True)
    with col_fig4:
        fig4_path = f"src/reports/figures/wall_hits_comparison_{size_key}.png"
        if os.path.exists(fig4_path):
            st.image(fig4_path, caption=f"Biểu đồ 4: So sánh Số lần va chạm tường trung bình mỗi Episode - Bản đồ {size_key}", use_container_width=True)

    # Hàng 3: Steps Boxplot và Radar Comparison
    st.markdown("---")
    col_fig5, col_fig6 = st.columns(2)
    with col_fig5:
        fig5_path = f"src/reports/figures/steps_boxplot_{size_key}.png"
        if os.path.exists(fig5_path):
            st.image(fig5_path, caption=f"Biểu đồ 5: Phân bố số bước đi khi thắng (Boxplot) - Bản đồ {size_key}", use_container_width=True)
    with col_fig6:
        fig6_path = f"src/reports/figures/radar_comparison_{size_key}.png"
        if os.path.exists(fig6_path):
            st.image(fig6_path, caption=f"Biểu đồ 6: Đánh giá hiệu năng toàn diện đa tiêu chí (Radar Chart) - Bản đồ {size_key}", use_container_width=True)
