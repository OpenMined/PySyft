<script>
  import { shortName } from '../lib/utils.js';
  export let user_info;
</script>

<main>
  <div id="nav-bar">
    <div id="bar-icons">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="rgb(25, 179, 230)"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        ><path
          d="M12 2c5.514 0 10 4.486 10 10s-4.486 10-10 10-10-4.486-10-10 4.486-10 10-10zm0-2c-6.627 0-12 5.373-12 12s5.373 12 12 12 12-5.373 12-12-5.373-12-12-12zm1.25 17c0 .69-.559 1.25-1.25 1.25-.689 0-1.25-.56-1.25-1.25s.561-1.25 1.25-1.25c.691 0 1.25.56 1.25 1.25zm1.393-9.998c-.608-.616-1.515-.955-2.551-.955-2.18 0-3.59 1.55-3.59 3.95h2.011c0-1.486.829-2.013 1.538-2.013.634 0 1.307.421 1.364 1.226.062.847-.39 1.277-.962 1.821-1.412 1.343-1.438 1.993-1.432 3.468h2.005c-.013-.664.03-1.203.935-2.178.677-.73 1.519-1.638 1.536-3.022.011-.924-.284-1.719-.854-2.297z"
        /></svg
      >
    </div>

    <div id="bar-icons">
      <svg width="24" height="24" xmlns="http://www.w3.org/2000/svg" fill="rgb(25, 179, 230)"
        ><path
          d="M22 9.74l-2 1.02v7.24c-1.007 2.041-5.606 3-8.5 3-3.175 0-7.389-.994-8.5-3v-7.796l-3-1.896 12-5.308 11 6.231v8.769l1 3h-3l1-3v-8.26zm-18 1.095v6.873c.958 1.28 4.217 2.292 7.5 2.292 2.894 0 6.589-.959 7.5-2.269v-6.462l-7.923 4.039-7.077-4.473zm-1.881-2.371l9.011 5.694 9.759-4.974-8.944-5.066-9.826 4.346z"
        /></svg
      >
    </div>

    <div id="bar-icons">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        fill="rgb(25, 179, 230)"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        ><path
          d="M15 21c0 1.598-1.392 3-2.971 3s-3.029-1.402-3.029-3h6zm.137-17.055c-.644-.374-1.042-1.07-1.041-1.82v-.003c.001-1.172-.938-2.122-2.096-2.122s-2.097.95-2.097 2.122v.003c.001.751-.396 1.446-1.041 1.82-4.668 2.709-1.985 11.715-6.862 13.306v1.749h20v-1.749c-4.877-1.591-2.193-10.598-6.863-13.306zm-3.137-2.945c.552 0 1 .449 1 1 0 .552-.448 1-1 1s-1-.448-1-1c0-.551.448-1 1-1zm-6.451 16c1.189-1.667 1.605-3.891 1.964-5.815.447-2.39.869-4.648 2.354-5.509 1.38-.801 2.956-.76 4.267 0 1.485.861 1.907 3.119 2.354 5.509.359 1.924.775 4.148 1.964 5.815h-12.903z"
        /></svg
      >
    </div>
    <a href="/profile"
      ><div id="user-profile-circle">
        <h3 style="color:white"><b> {shortName(user_info.name)} </b></h3>
      </div></a
    >
  </div>
</main>

<style>
  #nav-bar {
    position: fixed;
    top: 0;
    width: 85%;
    height: 5%;
    left: 15%;
    margin-top: 1%;
    display: flex;
    justify-content: right;
    align-items: center;
  }

  #bar-icons {
    margin-right: 25px;
    cursor: pointer;
  }
  #user-profile-circle {
    height: 5vh;
    cursor: pointer;
    width: 5vh;
    border-radius: 50%;
    margin-right: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: rgb(25, 179, 230);
  }
</style>
