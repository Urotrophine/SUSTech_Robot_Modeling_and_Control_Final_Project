% ==========================================================
% 4自由度(3R1P)机械臂 符号推导脚本 (包含工具坐标系与分离动力学)
% ==========================================================
clear; clc;

% ==========================================================
% 0. 初始化符号变量
% ==========================================================
fprintf('正在初始化符号变量...\n');

% 关节变量 (q1~q3为旋转角, q4为移动量) 及速度
syms q1 q2 q3 q4 real
syms dq1 dq2 dq3 dq4 real
q = [q1; q2; q3; q4];
dq = [dq1; dq2; dq3; dq4];

% 运动学结构参数 (DH参数: 连杆长度 a, 偏置 d, 扭转角 alpha)
syms a1 a2 a3 d1 d3 real
syms alpha2 alpha3 real 

% 本体动力学参数 (质量 m, 重力加速度 g)
syms m1 m2 m3 m4 g real
% 本体质心位置 (相对于各自连杆坐标系的坐标)
syms cx1 cy1 cz1 cx2 cy2 cz2 cx3 cy3 cz3 cx4 cy4 cz4 real
% 本体连杆主惯性矩 (对角线元素)
syms Ixx1 Iyy1 Izz1 Ixx2 Iyy2 Izz2 Ixx3 Iyy3 Izz3 Ixx4 Iyy4 Izz4 real

% ==========================================================
% 【新增】夹爪(Tool) 符号参数
% ==========================================================
% 工具坐标系相对于第4连杆的固定位置平移偏置
syms x_tool y_tool z_tool real
% 工具质量与质心位置(相对于工具坐标系)
syms m_tool real
syms cx_tool cy_tool cz_tool real
% 工具主惯性矩
syms Ixx_tool Iyy_tool Izz_tool real

% 标准 DH 变换矩阵的匿名函数
dh_matrix = @(theta, d, a, alpha) [
    cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta);
    sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta);
    0,           sin(alpha),             cos(alpha),            d;
    0,           0,                      0,                     1
];

% ==========================================================
% 1. 正运动学与工具坐标系及工具速度
% ==========================================================
fprintf('\n--- 1. 计算正运动学与工具速度 (Velocity Kinematics) ---\n');

% 1.1 本体 DH 变换
T1 = dh_matrix(q1, d1, a1, 0);
T2 = dh_matrix(q2, 0,  a2, alpha2);
T3 = dh_matrix(q3, d3, a3, alpha3);
T4_joint = dh_matrix(0, q4, 0, 0); % 第4关节为移动关节

T01 = T1;
T02 = T01 * T2;
T03 = T02 * T3;
T04 = T03 * T4_joint;

% 1.2 夹爪工具坐标系映射
% 通过常数变换矩阵 T_tool_local 将工具固连在执行器末端 (Link4)
T_tool_local = [eye(3), [x_tool; y_tool; z_tool]; 0 0 0 1];
T0_tool = T04 * T_tool_local;

% 提取全局工具中心点位置
P_tool = T0_tool(1:3, 4);

% 1.3 工具速度推导 (Velocity Kinematics)
% 工具线速度矢量 (基于末端位置对关节变量的雅可比计算得到)
Jv_tool = jacobian(P_tool, q);
V_tool_linear = Jv_tool * dq;

% 工具角速度矢量 (因为刚性固连，工具角速度即为末端角速度)
z0 = [0;0;1]; 
z1 = T01(1:3, 3); 
z2 = T02(1:3, 3); 
Jw_tool = sym(zeros(3,4));
Jw_tool(:,1) = z0;
Jw_tool(:,2) = z1;
Jw_tool(:,3) = z2;
% 第4关节为平动关节，不产生新的角速度，因此第4列为零
V_tool_angular = Jw_tool * dq;

fprintf('正运动学 T04、夹爪工具坐标系 T0_tool 及工具空间速度 V_tool 计算完成。\n');

% ==========================================================
% 2. 动力学方程 (分离建模：本体 + 工具)
% ==========================================================
fprintf('\n--- 2. 计算动力学方程 (分离建模) ---\n');

% ==========================================
% 2.1 本体部分建模 (Body: Link 1~4)
% ==========================================
Pc1_global = T01 * [cx1; cy1; cz1; 1]; Pc1 = Pc1_global(1:3);
Pc2_global = T02 * [cx2; cy2; cz2; 1]; Pc2 = Pc2_global(1:3);
Pc3_global = T03 * [cx3; cy3; cz3; 1]; Pc3 = Pc3_global(1:3);
Pc4_global = T04 * [cx4; cy4; cz4; 1]; Pc4 = Pc4_global(1:3);

m_vec = [m1, m2, m3, m4];
Pc_cell = {Pc1, Pc2, Pc3, Pc4};
I_cell = {diag([Ixx1, Iyy1, Izz1]), diag([Ixx2, Iyy2, Izz2]), diag([Ixx3, Iyy3, Izz3]), diag([Ixx4, Iyy4, Izz4])};

% 本体势能
P_body = 0;
for i = 1:4
    P_body = P_body + m_vec(i) * g * Pc_cell{i}(3); 
end

% 本体质量矩阵
M_body = sym(zeros(4,4));
for i = 1:4
    Jvi = jacobian(Pc_cell{i}, q);
    Jwi = sym(zeros(3,4));
    if i >= 1, Jwi(:,1) = z0; end
    if i >= 2, Jwi(:,2) = z1; end
    if i >= 3, Jwi(:,3) = z2; end
    
    if i==1, Ri = T01(1:3,1:3); 
    elseif i==2, Ri = T02(1:3,1:3); 
    elseif i==3, Ri = T03(1:3,1:3); 
    else, Ri = T04(1:3,1:3); end
    
    I_global = Ri * I_cell{i} * Ri.';
    M_body = M_body + m_vec(i) * (Jvi.' * Jvi) + (Jwi.' * I_global * Jwi);
end

% ==========================================
% 2.2 夹爪工具单独建模 (Tool)
% ==========================================
% 工具质心在全局坐标系下的表达
Pc_tool_local = [cx_tool; cy_tool; cz_tool; 1];
Pc_tool_global = T0_tool * Pc_tool_local; 
Pc_tc = Pc_tool_global(1:3);

% 工具单独产生的势能
P_tool_pot = m_tool * g * Pc_tc(3);

% 工具单独的质量矩阵贡献
Jv_tc = jacobian(Pc_tc, q);
Jw_tc = Jw_tool; % 夹爪与Link4固连，其角速度雅可比与末端保持一致
R_tool_global = T0_tool(1:3, 1:3);
I_tool_global = R_tool_global * diag([Ixx_tool, Iyy_tool, Izz_tool]) * R_tool_global.';

M_tool = m_tool * (Jv_tc.' * Jv_tc) + (Jw_tc.' * I_tool_global * Jw_tc);

% ==========================================
% 2.3 融合总动力学方程 (本体 + 工具)
% ==========================================
% 融合总势能与重力矩阵
P_total = P_body + P_tool_pot;
G = jacobian(P_total, q).';
fprintf('总重力矩阵 G(q) (本体+工具) 计算完成。\n');

% 融合总质量矩阵
M = M_body + M_tool;
fprintf('总质量矩阵 M(q) (本体+工具) 计算完成。\n');

% 计算总科氏力矩阵
fprintf('正在计算总科氏力矩阵 C(q,dq) (符号求导较慢，请耐心等待)...\n');
C = sym(zeros(4,4));
for k = 1:4
    for j = 1:4
        for i = 1:4
            c_ijk = 0.5 * (diff(M(k,j), q(i)) + diff(M(k,i), q(j)) - diff(M(i,j), q(k)));
            C(k,j) = C(k,j) + c_ijk * dq(i);
        end
    end
end
fprintf('总科氏力矩阵 C(q,dq) 计算完成。\n');

% ==========================================================
% 3. 逆运动学 (Inverse Kinematics) 符号提取
% ==========================================================
fprintf('\n--- 3. 逆运动学方程 (IK) 提取 ---\n');
Px_tool = P_tool(1);
Py_tool = P_tool(2);
Pz_tool = P_tool(3);

fprintf('逆运动学符号方程 (Px_tool, Py_tool, Pz_tool) 已提取。\n');
disp('=== 所有符号推导完毕！===');


