<!DOCTYPE html>
<html> 
    <head>
        <link rel="stylesheet" href="style.css">
        <title> Pricing Sugguestion </title>
    </head>
    <body>
        <br> <br> <br> <br> <br> <br> <br> 
        <h1> Price Suggestion! </h1>
        <form action="price.php" method="post">
        <table class="content-table">
            <thead>
                <tr>
                    <th>Item Details</th>
                    <th>Input</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="active-data">Item Name</td>
                    <td>  <input type ='text' id ='itemname' style = 'width: 200px' name='name'>   </td>
                </tr>
                <tr>
                    <td class="active-data">Item Condition</td>
                    <td> <select id="condition" style = 'width: 208px' name='item_condition_id' >
                        <option value="1">New</option>
                        <option value="2">Like New</option>
                        <option value="3">Good</option>
                        <option value="4">Fair</option>
                        <option value="5">Poor</option>
                      </select>  </td>
                </tr>
                <tr>
                    <td class="active-data">Category</td>
                    <td> <input type ='text' id ='category' style = 'width: 200px' name='category'> </td>
                </tr>
                <tr>
                    <td class="active-data">Brand</td>
                    <td> <input type ='text' id ='brand' style = 'width: 200px' name='brand'>  </td>
                </tr>
                <tr>
                    <td class="active-data">Shipping</td>
                    <td> <input type="radio" id="shipping" name="shipping" value="1">
                        <label for="Yes">Yes</label>
                        
                        <input type="radio" id="shipping" name="shipping" value="0">
                        <label for="No">No</label><br></td>
                </tr>
                <tr>
                    <td class="active-data">Item Description</td>
                    <td> <input type ='text' id ='category' style = 'width: 200px' name='item_description'> </td>
                </tr>
            </tbody>
        </table>
        
        <input type="submit" value="Submit">
        </form>
        <?php
            if ($_SERVER["REQUEST_METHOD"] == "POST") {
            // collect value of input field

            $post = [
                'name' => $_POST['name'],
                'item_condition_id' => $_POST['item_condition_id'],
                'category' => $_POST['category'],
                'brand' => $_POST['brand'],
                'shipping' => $_POST['shipping'],
                'item_description' => $_POST['item_description'],
            ];
            $ch = curl_init('localhost:5000/getPrice');
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
            curl_setopt($ch, CURLOPT_POSTFIELDS, $post);
            
            // execute!
            $response = curl_exec($ch);
            
            // close the connection, release resources used
            curl_close($ch);
            
            // do anything you want with your response
            echo "<h1> The predicted price is <u> $ {$response} </u> </h1>";
            }
        ?>
    </body>
</html>
