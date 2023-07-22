$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        console.log('Success1!');
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            
            contentType: false,
            cache: false,
            processData: false,
            async: false,
            success: function (data) {
                
                // console.log('Success2!');
                // // // Get and display the result
                // // window.location.href = 'http://127.0.0.1:5000/result/'+data +".html";
                // location.replace('http://127.0.0.1:5000/result/'+data +".html");
                // console.log('Success!');

                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(data);
                console.log('Success!');
                // window.location.href = "/result/" + data;
            },
        
        });
    });

});
