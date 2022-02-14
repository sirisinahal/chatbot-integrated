$(document).ready(function() {
    $("#btnFetch1").click(function(event){
        <b>alert($(this).prop("value"));</b>
       });
        
    $("btnFetch2").click(function(event){
        <b>alert($(this).prop("name"));</b>
       });



    $("#btnFetch").click(function() {
    var text = $(this).attr('value'); 
    alert(text);
    // disable button
    $(this).prop("disabled", true);
    // add spinner to button
    $(this).html(
    '<i class="fa fa-circle-o-notch fa-spin"></i> loading...'
    );
    $.get('/typebot',function(data){
        // $(location).prop('href', 'index.html');
        // var x='<form action='
        console.log(data)
        window.location.replace('http://127.0.0.1:5000/indexfile');
      
    });
    return false; 
    // $.ajax({
    //     type: "GET",
    //     url: "Default.aspx/GetResponse",
    //     data: '{}',
    //     contentType: "application/json; charset=utf-8",
    //     dataType: "json",
    //     success: function (response) {
    //        if (response.d == true) {
    //             alert("You will now be redirected.");
    //             window.location = "//www.aspsnippets.com/";
    //         }
    //     },
    //     failure: function (response) {
    //         alert(response.d);
    //     }
   

    });
    });
    

//     $.ajax({
//         url: "/user",
//         type: "POST",
//         data: JSONobj,
//         dataType: "json",
//         success: function() {
//             window.location.href = "main.html";
//         },
//         contentType: "application/json"
// });