(() => {
  // ns-params:@params
  var disableToBottom = false;
  var disableToTop = false;

  // <stdin>
  var toTopButton = document.getElementById("to-top-button");
  toTopButton && toTopButton.addEventListener("click", goToTop);
  var toBottomButton = document.getElementById("to-bottom-button");
  toBottomButton && toBottomButton.addEventListener("click", goToBottom);
  window.onscroll = function() {
    if (!disableToTop) {
      if (document.body.scrollTop > 500 || document.documentElement.scrollTop > 500) {
        toTopButton.classList.remove("hidden");
      } else {
        toTopButton.classList.add("hidden");
      }
    }
    if (!disableToBottom) {
      if (isBottom()) {
        toBottomButton.classList.add("hidden");
      } else {
        toBottomButton.classList.remove("hidden");
      }
    }
  };
  function goToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  function goToBottom() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }
  function isBottom() {
    let clientHeight = document.documentElement.clientHeight;
    let scrollHeight = document.documentElement.scrollHeight;
    let scrollTop = document.documentElement.scrollTop;
    let scrollHeightMinusClientHeight = scrollHeight - clientHeight;
    if (scrollHeightMinusClientHeight <= scrollTop) {
      return true;
    } else {
      return false;
    }
  }
})();
