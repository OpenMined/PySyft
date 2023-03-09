<div class="wrapper">
  <div class="circle">
    <div class="checkMark" />
  </div>
</div>

<style lang="postcss">
  .wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 30px;
  }

  .circle {
    position: relative;
    background: #00b01d;
    border-radius: 100%;
    height: 30px;
    width: 30px;
  }

  .checkMark {
    position: absolute;
    transform: rotate(50deg) translate(-50%, -50%);
    left: 27%;
    top: 43%;
    height: 15px;
    width: 8px;
    border-bottom: 3px solid #fff;
    border-right: 3px solid #fff;
  }
</style>
