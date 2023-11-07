# Use an official Node.js runtime as the base image
FROM node:17-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json for npm install
COPY package.json .

# Install app dependencies
RUN npm install

# Copy the entire React app directory into the container
COPY . .

# Expose the port where the app will run (if necessary)
EXPOSE 3000

# Start the React development server
CMD [ "npm", "start" ]