---
layout: default
title: Projects
---

Is this printing?
<div class="posts">
  {% for post in paginator.posts %}
  <hr>
  <article class="post">
    <h1 class="post-title">
      <a href="{{ site.baseurl }}{{ post.url }}">
        {{ post.title }}
      </a>
    </h1>

    <time datetime="{{ post.date | date_to_xmlschema }}" class="post-date">{{ post.date | date_to_string }}</time>

   <!-- {{ post.excerpt}} -->
    <em>{{post.sub-title}}</em>
  </article>
  {% endfor %}
</div>
<!--
<div class="pagination">
  {% if paginator.next_page %}
    <a class="pagination-item older" href="{{ paginator.next_page_path | prepend: site.baseurl }}">Older</a>
  {% else %}
    <span class="pagination-item older">Older</span>
  {% endif %}
  {% if paginator.previous_page %}
    <a class="pagination-item newer" href="{{ paginator.previous_page_path | prepend: site.baseurl }}">Newer</a>
  {% else %}
    <span class="pagination-item newer">Newer</span>
  {% endif %}
</div>
-->
<div class="pagination">
  {% if paginator.total_pages > 1 %}
  <ul class="pager main-pager">
    {% if paginator.previous_page %}
    <h6 class="previous">
      <a href="{{ paginator.previous_page_path | prepend: site.baseurl | replace: '//', '/' }}">&larr; Newer Posts</a>
    </h6>
    {% endif %}
    {% if paginator.next_page %}
    <h6 class="next">
      <a href="{{ paginator.next_page_path | prepend: site.baseurl | replace: '//', '/' }}">Older Posts &rarr;</a>
    </h6>
    {% endif %}
  </ul>
  {% endif %}
</div>
