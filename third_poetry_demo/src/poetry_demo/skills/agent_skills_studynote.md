@TOC

# 【README】

1. 本文总结自B站《agent skill ，从使用到原理，马克的技术工作坊》
2. 本文内容包括：
   1. skills 概念；
   2. 基本用法；
   3. 高级用法， 包括Reference， script； 
   4. 与mcp比较； 

<br>

---

# 【1】Agent Skill 概念

1. Agent Skill：是大模型可以随时翻阅的说明文档；
   1. 【例】请将会议内容总结为如下几点：
      1. 参会人员；
      2. 议题；
      3. 决定；

<br>

# 【2】Agent Skill基本用法

## 【2.1】创建skill

1. 在用户目录文件夹，创建/.claude/skill 
2. 创建文件夹，会议总结助手；

```shell
tom@TomMacbook %1~ %# mkdir skill
tom@TomMacbook %1~ %# cd skill
tom@TomMacbook %1~ %# mkdir 会议总结助手
```

接着使用VS Code打开这个文件夹【会议总结助手】；

<br>

在【会议总结助手】文件夹中新建 SKILL.md文件，如下。

<font color=red>SKILL.md 文档结构</font>：

1. 元数据（--- 与 ---之间的内容）；
2. 指令（# 会议总结助手到 输出）；

```markdown
---
name: 会议总结助手
description: 该技能用于根据会议录音总结内容
---

# 会议总结助手

## 总结规则

请将会议内容总结为以下几点：

- 参会人员
- 议题
- 决定

注意：每项都只能分别使用一句话来表述，不要分成多条。

## 示例

输入：

张三：那我们开始吧，今天主要是把下个月社区志愿活动的安排一次性定下来。
李四：我建议活动放在公园，人多也方便组织。
王五：可以，不过要提前申请场地，不然可能有风险。
赵六：场地申请我可以负责，这周内给大家结果。
孙七：人数最好先有个范围，方便准备物资。
张三：那就先按 50 人左右来估算吧。
李四：上次的手套还能用，但垃圾袋需要再买。
王五：预算要不要设个上限，避免超支。
张三：预算控制在 1000 以内，优先用现有物资。
孙七：时间我建议周六上午，天气也不会太热。
李四：九点集合应该比较合适。
赵六：我周三前把申请结果同步到群里。
张三：好，那报名截止时间定在周四晚上。
王五：周五可以统一分组和采购。
孙七：我来负责写报名文案和活动当天的合影安排。
张三：安全方面提醒大家带水，活动结束简单总结一下就行。
张三：那今天就到这，大家按分工推进吧。

输出：

- 参会人员：张三、李四、王五、赵六、孙七
- 议题：统一确定下个月社区志愿活动的地点、时间、人数、预算及分工安排。
- 决定：活动定在公园并于周六上午九点举行，按约 50 人规模和 1000 预算执行，由赵六负责场地申请、孙七负责宣传及合影，其余成员配合物资和分组。

```

【文档解说】



接着随便打开一个文件夹，打开claude code；

![skill_01](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/skills/img/skill_02.png)

<br>

![skill_03](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/skills/img/skill_03.png)

<br>

---

### 【2.1.1.】用户与CluadeCode与大模型交互时序图

![skill_04](/Users/rong/studynote/workbench/python/third_poetry_demo/src/poetry_demo/skills/img/skill_04.png)

<br>

### 【2.1.2】skill的第一个核心机制-按需加载

1. agent skill的第1个核心机制：<font color=red>按需加载</font>；
   1. 虽然skill的名字和描述是始终对模型可见的；
   2. 但具体的指令内容，只有在这个skill被模型选中后，才会被加载进来给模型看；这就节省了很多token了；

<br>

---

## 【2.2】Agent Skill的高级用法（Reference篇）



























