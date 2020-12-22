function [h, display_array] = displayData(X, example_width)
#   [h, display_array] = DISPLAYDATA(X, example_width) 
#   Visualizes a subset of 100 digits from the MNIST dataset stored in X
#   Each row of X represents a 20 by 20 pixel grayscale image of a digit
#
#   This function returns the figure handle h and the displayed array

# Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
end

# Compute the rows and columns of the input matrix X
[m n] = size(X);

# Compute the height of a single image
example_height = (n / example_width);

# Compute number of images to display in each row and column
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

# Set the current colomap to gray (input images are in grayscale)
colormap(gray);

# Between images padding
pad = 1;

# Setup a 2D display array
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

# Copy each image into a patch of the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m
			break; 
		endif
		# Get the maximum grayscale shade value of each image
		max_val = max(abs(X(curr_ex, :)));
    # Copy the image to the display array, normalized against max shade value
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	endfor
	if curr_ex > m
		break; 
	endif
endfor

# Create the figure handle h
# 'imagesc' displays a scaled version of the input matrix as a color image:
# 1. The colormap is automatically scaled so that the entries of the matrix occupy 
#    the entire colormap
# 2. The second argument [-1 1] sets the limits of the 'clim' property of the 
#    color axis. Default is [0 1]
h = imagesc(display_array, [-1 1]);

# Setup the axes and update the figure
# - Option 'image' does two things:
#    1. It forces x-axis unit distance to equal y-axis unit distance
#    2. It fixes the axes to the limits of the data 
# - Option 'off' hides the axes
axis image off
drawnow;

endfunction