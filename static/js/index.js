window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    bulmaSlider.attach();

    // Before/after comparison slider
    var baSlider = document.getElementById('ba-slider');
    if (baSlider) {
      var baBefore = document.getElementById('ba-before');
      var baLine = document.getElementById('ba-line');
      var dragging = false;
      var setPos = function(clientX) {
        var rect = baSlider.getBoundingClientRect();
        var p = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        var clip = 'inset(0 ' + ((1 - p) * 100) + '% 0 0)';
        baBefore.style.clipPath = clip;
        baBefore.style.webkitClipPath = clip;
        baLine.style.left = (p * 100) + '%';
      };
      baSlider.addEventListener('pointerdown', function(e) {
        dragging = true;
        baSlider.setPointerCapture(e.pointerId);
        setPos(e.clientX);
      });
      baSlider.addEventListener('pointermove', function(e) {
        if (dragging) setPos(e.clientX);
      });
      baSlider.addEventListener('pointerup', function() { dragging = false; });
      baSlider.addEventListener('pointercancel', function() { dragging = false; });
    }

})
