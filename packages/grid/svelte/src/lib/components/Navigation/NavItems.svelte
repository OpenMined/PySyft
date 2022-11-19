<script context="module">
  import { faUserPlus, faSignIn, faUsers } from '@fortawesome/free-solid-svg-icons';

  export const navItems = [
    { page: 'Login', slug: '/login', icon: faSignIn },
    { page: 'Signup', component: '/signup', icon: faUserPlus },
    { page: 'Users', component: '/users', icon: faUsers }
  ];
</script>
