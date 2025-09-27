(() => {
  // ns-params:@params
  var enableCounts = true;
  var envId = "https://sensational-kataifi-4ebf86.netlify.app/.netlify/functions/twikoo";
  var lang = "en";
  var path = "";
  var region = "";

  // <stdin>
  twikoo.init({
    envId,
    el: "#twikoo-comment",
    region: region || "ap-guangzhou",
    path: path || location.pathname,
    lang: lang || "en"
  });
  if (enableCounts) {
    twikoo.getCommentsCount({
      envId,
      region: region || "ap-guangzhou",
      urls: [path || location.pathname],
      includeReply: false
    }).then(function(res) {
      let commentCount = document.getElementById("speed-comment-count-id");
      res.forEach((element) => {
        if (commentCount.innerText === "Comment") {
          commentCount.innerHTML = `${element.count} Comments`;
        } else {
          return;
        }
      });
    }).catch(function(err) {
      console.error(err);
    });
  }
})();
