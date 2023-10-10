<script>
  export let link;
</script>

<a href="#{link.toLowerCase()}">{link}</a>

<style>
  a {
    color: black;
    padding: 12px 16px;
    text-decoration: none;
    display: block;
  }

  a:hover {
    background-color: #ddd;
  }
</style>
