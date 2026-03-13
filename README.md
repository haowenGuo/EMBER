flowchart TD
    %% 定义节点样式（适配学术排版）
    classDef start fill:#f9f,stroke:#333,stroke-width:1px
    classDef react fill:#9ff,stroke:#333,stroke-width:1px
    classDef eval fill:#ff9,stroke:#333,stroke-width:1px
    classDef endnode fill:#9f9,stroke:#333,stroke-width:1px

    %% 流程节点
    A[对话启动]:::start --> B[接收对抗输入+历史语境]
    B --> C[LLM生成初始输出]
    
    %% REACT 核心闭环
    C --> D[思考模块 Reason]:::react
    D --> D1[语义分析+偏见识别<br/>制定修正策略]
    D1 --> E[执行模块 Act]:::react
    E --> E1[删除偏见表达<br/>中立化修改<br/>补充多视角]
    E1 --> F[生成初步修正输出]
    F --> G[观察模块 Observe]:::react
    G --> G1[捕捉修正效果<br/>记录环境反馈]
    G1 --> H[反思模块 Reflect]:::react
    H --> H1[对比预期vs实际<br/>分析策略不足<br/>更新经验库]
    
    %% 评估与迭代
    H1 --> I[偏见评估模块]:::eval
    I --> I1[量化偏见指标<br/>生成评估结果]
    I1 --> J[输出最终无偏结果]:::endnode
    J --> K{对话是否结束?}
    K -- 否 --> B[接收对抗输入+历史语境]
    K -- 是 --> L[对话终止]:::endnode

    %% 辅助关联线（体现数据流转）
    I1 -. 历史评估结果 .-> D[思考模块 Reason]
    H1 -. 反思经验 .-> D[思考模块 Reason]
