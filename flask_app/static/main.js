$(function() {

var leftKeyCodes = [49, 50, 51, 52, 53, 81, 87, 69, 82, 84, 65, 83, 68, 70, 71, 90, 88, 67, 86, 66, 18, 9, 16, 17, 20, 27];
var rightKeyCodes = [54, 55, 56, 57, 48, 89, 85, 72, 73, 74, 75, 76, 77, 78, 79, 80, 13, 8, 46, 40, 37, 38, 39, 107, 108, 109, 110, 111, 188, 186, 187, 189, 190, 191, 219, 220, 221, 222];
var prompts = ["Describe what you did last weekend.", 
               "Describe your childhood home.", 
               "Describe the room you are sitting in right now.", 
               "Describe your favorite activity and what you like about it.", 
               "Describe your favorite season of the year.", 
               "Describe your day."
               ];
var fullData = [];
        
function display(mills) {
    if (mills > 1000)
        return (mills / 1000) + ' s';
    return mills + ' ms';
}

function submitAjaxRequest(json_data) {
    $("#submit").click(function(e) {
        console.log(JSON.stringify(json_data));
        var gender = $("#gender_select option:selected").text();
        var tremors = $("#tremors_select option:selected").text();
        var sided = $("#sided_select option:selected").text();
        json_data.unshift({
            "Gender": gender, 
            "Tremors": tremors, 
            "Sided": sided
        });
        $.ajax({
            type: 'POST',
            url: '/results',
            data: JSON.stringify(json_data),
            dataType: 'json',
            contentType: 'application/json; charset=utf-8',
            success: function(response) {
                console.log(response);
            },
            error: function(request, error) {
                console.log(error);
            }
        });
    });
}
// Format of rows: [hand(left or right), hold time, direction (LR, RL, SS), latency time (keydown-keydown), flight time (keyup-keydown)]
function init() {

    var dwellTimes = {};
    var currentKeyPressed = 0;
    var currentHoldTime = 0;
    var latencyDateTime = 0;
    var flightDateTime = 0;
    var lastKeyPressed = 'L';
    var counter = 0;
    var fired = false;
    $('#in').keydown(function(e) {
        if(!fired) {
            fired = true;
            var keyCode = e.which;
            var latency = 0;
            
            if (!dwellTimes[keyCode])
                dwellTimes[keyCode] = new Date().getTime();
                //console.log(dwellTimes[keyCode]);
                
            if(counter % 2 == 0) {
                latencyDateTime = new Date().getTime();
            }
            
            if (counter % 2 == 1) {
                latency = new Date().getTime() - latencyDateTime;
                //console.log('Current datetime: ' + latencyDateTime);
                //console.log('Latency: ' + latency);
                flight = new Date().getTime() - flightDateTime;
                //console.log('Flight: ' + flight);
                var secondKey = "L";
                if (leftKeyCodes.includes(keyCode)) {
                   secondKey = "L";
                } else if (rightKeyCodes.includes(keyCode)) {
                    secondKey = "R";
                } else if (keyCode == 32) {
                    secondKey = "S";
                }
                direction = lastKeyPressed + secondKey;
                //console.log("Last key: " + lastKeyPressed);
                //console.log("Second key: " + secondKey);
                //console.log(direction);
                
                rowToAdd = {
                    "Hand": lastKeyPressed, 
                    "Hold time": currentHoldTime, 
                    "Direction": direction, 
                    "Latency time": latency, 
                    "Flight time": flight
                };
                
                fullData.push(rowToAdd);
                //console.log(fullData);
                //currentRow = [];
                //currentRow.push(latency);
            }
            
            if (leftKeyCodes.includes(keyCode)) {
                  //console.log("Left key!");
                lastKeyPressed = "L";
                //currentRow.push("L");
            } else if (rightKeyCodes.includes(keyCode)) {
                  //console.log("Right key");
                lastKeyPressed = "R";
                //currentRow.push("R");
            } else if (keyCode == 32) {
                  //console.log("Space!");
                lastKeyPressed = "S";
                //currentRow.push("S");
            }
            
            counter++;
        }
       
    }).keyup(function(e) {
          //console.log(e.which);
          var keyCode = e.which;
          var dwellTime = new Date().getTime() - dwellTimes[keyCode];
          
          currentHoldTime = dwellTime;
          delete dwellTimes[keyCode];
          
          flightDateTime = new Date().getTime();
          //console.log('Flight date time: ' + flightDateTime);
          //$('#output').prepend("<p>Pressed key " + e.which + " for " + dwellTime / 1000 + "</p>");
          //console.log(currentRow);
          fired = false;
    });
}

init();
submitAjaxRequest(fullData);

});