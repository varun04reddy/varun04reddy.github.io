---
layout: page
permalink: /publications/
title: publications
nav: true
years: [2024]

nav_order: 2
---

<!-- _pages/publications.md -->
<div class="publications">

<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
