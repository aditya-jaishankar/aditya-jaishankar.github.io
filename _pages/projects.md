---
title: "List of Projects by Tag"
permalink: /projects/
author_profile: true
layout: single
---

{% assign tags =  site.projects | map: 'tags' | join: ','  | split: ',' | uniq %}
{% for tag in tags %}
  <h3>{{ tag }}</h3>
  <ul>
  {% for project in site.projects %}
    {% if project.tags contains tag %}
    <li><a href="{{ site.baseurl }}{{ project.url }}">{{ project.title }}</a></li>
    <i>{{ project.excerpt }}</i>
    {% endif %}
  {% endfor %}
  </ul>
{% endfor %}