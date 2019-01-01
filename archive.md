---
layout: page
title: Archive
---

## Projects

<!--
{% for post in site.posts %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ post.url }})
{% endfor %}
-->

{% assign posts = site.posts | where_exp: "post", "post.categories contains 'project'" %}
{% for post in posts %}
  <li>{{ post.date | date_to_string }} &raquo;<a href="{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}

## Blog posts

{% assign posts = site.posts | where_exp: "post", "post.categories contains 'blog'" %}
{% for post in posts %}
  <li>{{ post.date | date_to_string }} &raquo;<a href="{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}