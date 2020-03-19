# Import necessary modules; CSV is for reading .csv files and Plots is for plotting
using CSV
using Plots
using DataFrames
# Y = β_0 * X_0 + β_1 * X_1 + β_2 * X_2
# if X_0 = 1: we can write the above as Y = β * X for vectorization
# Predictions from Model β: Y_Pred = β * X

# Read the dataset from file
dataset = CSV.read("C:\\Users\\Avyakta\\github\\GNR-ML\\Assignment1\\data\\housingPriceData.csv")

# Extract columns from the dataset
dataset = DataFrame(dataset)
function partitionTrainValidationTest(data, at, at_1)
    n = nrow(data)
    idx = (1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    validation_idx = view(idx, (floor(Int, at*n)+1):(floor(Int, at_1*n)))
    test_idx = view(idx, (floor(Int, at_1*n)+1):n)
    data[train_idx,:], data[validation_idx,:], data[test_idx,:]
end
function standard(X)
    mean = sum(X)/length(X)
     var = sum((X .- (sum(X)/length(X))).^2) / (length(X))
     sd  = sqrt(var)
     Y = (X .- mean)/(sd)
    return Y
end
train,validation,test = partitionTrainValidationTest(dataset, 0.6,0.8)
#convert(Matrix, train)
course1 = train.price
course2 = standard(train.bedrooms)
course3 = standard(train.bathrooms)
course4 = standard(train.sqft_living)

#mean_sq_price = mean(course1)
#course1 = course1 .- mean_sq_price
#
# # Visualize the columns as a scatter plot
#display(scatter3d(course2, course3, course4))
#
# # Stub column 1 for vectorization.
m = length(course1)
x0 = ones(m)
# # Define the features array
X = cat(x0, course2, course3, course4, dims=2)
# column wise concatenation => dims=2
# Get the variable we want to regress
Y = course1

# Define a function to calculate cost function
function costFunction(X, Y, B, L)
    m = length(Y)
    cost = (sum(((X*B) - Y).^2))/(m) + L*sum((B[2:4,1]'*B[2:4,1]))/m
    return cost
end

# # Initial coefficients
B = zeros(4, 1)
L = 0
# Calcuate the cost with intial model parameters B=[0,0,0]
trial = X*B
intialCost = costFunction(X, Y, B, L)


# Define a function to perform gradient descent
function gradientDescent(X, Y, B, learningRate, numIterations, L)
    costHistory = zeros(numIterations)
    m = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        gradient = (X' * loss)/m + L*(B)/m
        gradient[1] = gradient[1] - L*(B[1])/m
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X, Y, B, L)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B, costHistory
end

#
learningRate = 0.3
lambda = 400
newB, costHistory = gradientDescent(X, Y, B, learningRate, 1000, lambda)

################################################################
course1_validation = validation.price
course2_validation = standard(validation.bedrooms)
course3_validation = standard(validation.bathrooms)
course4_validation = standard(validation.sqft_living)
# Make predictions using the learned model; newB
m_val = length(course1_validation)
x0_val = ones(m_val)

# # Define the features array
X_val = cat(x0_val, course2_validation, course3_validation , course4_validation, dims=2)
# column wise concatenation => dims=2
# Get the variable we want to regress
Y_val = course1_validation

YPred = X_val * newB

rmse_val = ((sum((YPred.-Y_val).^2))/m_val)^0.5
#print(rmse_val," ")

r_sq_val = 1 - (sum((YPred.-Y_val).^2))/(sum((Y_val.-(sum(Y_val)/length(Y_val))).^2))

################################################################
course1_test = test.price
course2_test = standard(test.bedrooms)
course3_test = standard(test.bathrooms)
course4_test = standard(test.sqft_living)
# Make predictions using the learned model; newB
m_test = length(course1_test)
x0_test = ones(m_test)
#
# # Define the features array
X_test = cat(x0_test, course2_test, course3_test, course4_test, dims=2)
# column wise concatenation => dims=2
# Get the variable we want to regress
Y_test = course1_test

YPred_test = X_test * newB

# visualize and compare the the prediction with original; below we plot only first 10 entries; plot! is to plot on the existing plot window
#display(plot(Y[1:1000]))
#display(plot!(YPred[1:1000]))

# Visualize the learning: how the loss decreased.
#display(plot(costHistory))

rmse_test = ((sum((YPred_test.-Y_test).^2))/m_test)^0.5

r_sq_test = 1 - (sum((YPred_test.-Y_test).^2))/(sum((Y_test.-(sum(Y_test)/length(Y_test))).^2))

#print(rmse_test)

df = DataFrame(YPred_test)
CSV.write("C:\\Users\\Avyakta\\github\\GNR-ML\\Assignment1\\data\\2a.csv", df)
