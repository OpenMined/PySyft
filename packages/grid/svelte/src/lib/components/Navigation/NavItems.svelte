<script context="module">
  import { faUserPlus, faSignIn, faHome, faUsers } from '@fortawesome/free-solid-svg-icons';

  export const navItems = [
    { title: 'Home', slug: '/', icon: faHome },
    { title: 'Login', slug: 'login', icon: faSignIn },
    { title: 'Signup', slug: 'signup', icon: faUserPlus },
    { title: 'Users', slug: 'users', icon: faUsers }
  ];
</script>
