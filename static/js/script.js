function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#image')
                .attr('src', e.target.result);
        };

        reader.readAsDataURL(input.files[0]);
    }
}

function previewImage() {
    // document.getElementById("image");
    var reader = new FileReader();
    reader.readAsDataURL(document.getElementById("image-source").files[0]);

    reader.onload = function (oFREvent) {
        document.getElementById("image").src = oFREvent.target.result;
    };

};


function resultImage() {
    document.getElementById("image2").src = "http://127.0.0.1:5500/scaling/foto10.JPG"
    Cache.delete()
};


document.addEventListener("DOMContentLoaded", function () {
    var elements = document.getElementsByTagName("INPUT");
    for (var i = 0; i < elements.length; i++) {
        elements[i].oninvalid = function (e) {
            e.target.setCustomValidity("");
            if (!e.target.validity.valid) {
                e.target.setCustomValidity("Data ini wajib diisikan, mohon diisi!");
            }
        };
        elements[i].oninput = function (e) {
            e.target.setCustomValidity("");
        };
    }
})