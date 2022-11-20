<script context="module">
  import { faUserPlus, faSignIn, faUsers, faHome } from '@fortawesome/free-solid-svg-icons';
  import Login from '../../../routes/Login/+page.svelte';
  import Signup from '../../../routes/Signup/+page.svelte';
  import Home from '../../../routes/+page.svelte';

  export const navItems = [
    { title: 'Home', slug: '/', icon: faHome, component: Home },
    { title: 'Login', slug: 'login', icon: faSignIn, component: Login },
    { title: 'Signup', slug: 'signup', icon: faUserPlus, component: Signup }
  ];
</script>
