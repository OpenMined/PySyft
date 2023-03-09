<script>
  import { Route } from 'svelte-routing';
  import Login from '../../routes/login/+page.svelte';
  import Signup from '../../routes/signup/+page.svelte';
  import Home from '../../routes/(app)/home/+page.svelte';
  import Users from '../../routes/(app)/users/+page.svelte';
</script>

<div class="w-full">
  <Route path="login" component={Login} />
  <Route path="signup" component={Signup} />
  <Route path="users" component={Users} />
  <Route path="/" component={Home} />
</div>
