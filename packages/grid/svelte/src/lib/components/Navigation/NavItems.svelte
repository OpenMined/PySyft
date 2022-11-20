<script context="module">
  import { faUserPlus, faSignIn, faUsers, faHome } from '@fortawesome/free-solid-svg-icons';
  import Login from '../../../routes/login/+page.svelte';
  import Signup from '../../../routes/signup/+page.svelte';
  import Home from '../../../routes/+page.svelte';

  export const navItems = [
    { title: 'Home', slug: '/', icon: faHome, component: Home },
    { title: 'Login', slug: 'Login', icon: faSignIn, component: Login },
    { title: 'Signup', slug: 'Signup', icon: faUserPlus, component: Signup }
  ];
</script>
