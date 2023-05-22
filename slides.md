---
theme: seriph
background: https://source.unsplash.com/collection/94734566/1920x1080
class: text-center
highlighter: shiki
lineNumbers: true
info: |
  # 这里写有关系的内容
drawings:
  persist: false
transition: slide-left
css: unocss
title: 这里是封面标题
---

<!-- 封面页 -->
# 这里是封面标题

<!-- 这里是正文，会被渲染成p标签 -->

<div class="pt-12" style="padding-top:0;">
  江西财经大学 | 易亚伟 | 2202291160@stu.edu.cn
</div>

---
layout: center
class: text-center
---

# 一、章节过渡

[Documentations](https://sli.dev) · [GitHub](https://github.com/slidevjs/slidev) · [Showcases](https://sli.dev/showcases.html)

---
transition: fade-out
---

# 列表-列出来一些东西

Slidev is a slides maker and presenter designed for developers, consist of the following features

- 📝 **Text-based** - focus on the content with Markdown, and then style them later
- 🎨 **Themable** - theme can be shared and used with npm packages
- 🧑‍💻 **Developer Friendly** - code highlighting, live coding with autocompletion
- 🤹 **Interactive** - embedding Vue components to enhance your expressions
- 🎥 **Recording** - built-in recording and camera view
- 📤 **Portable** - export into PDF, PNGs, or even a hostable SPA
- 🛠 **Hackable** - anything possible on a webpage

<!-- 换行直接用br标签 -->
<br>
<br>

Read more about [Why Slidev?](https://sli.dev/guide/why)

<!--
可以直接用css改当前页的样式
-->

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

<!--
最后一部分的注释会被当做ppt里的注释
-->

---
transition: slide-up
---

# 一级标题

Hover on the bottom-left corner to see the navigation's controls panel, [learn more](https://sli.dev/guide/navigation.html)
## 二级标题

### 三级标题

|  表头   |     |
| --- | --- |
| <kbd>right</kbd> / <kbd>space</kbd>| next animation or slide |
| <kbd>left</kbd>  / <kbd>shift</kbd><kbd>space</kbd> | previous animation or slide |
| <kbd>up</kbd> | previous slide |
| <kbd>down</kbd> | next slide |


---
layout: image-right
image: https://source.unsplash.com/collection/94734566/1920x1080
---
<!-- 右侧图片布局 -->

# 右侧图片布局

默认H1标签下面的一行会变灰

<!-- 设置代码展示顺序 -->
```ts {all|2|1-6|9|all}
interface User {
  id: number
  firstName: string
  lastName: string
  role: string
}

function updateUser(id: number, update: User) {
  const user = getUser(id)
  const newUser = { ...user, ...update }
  saveUser(id, newUser)
}
```

---

# 双列布局

这里可以放一些解释说明
<div grid="~ cols-2 gap-4">
<div>

### 这是一列


</div>
<div>

### 这是一列

</div>
</div>

<style>
h1{
  margin-top:0px;
}
</style>



