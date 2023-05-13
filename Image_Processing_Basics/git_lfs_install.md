## Git LFS Install for .ubyte Files

#### Uploading Large Binary Files to GitHub using Git LFS
  If you are working with large binary files in your Git repository, such as image or video files, uploading them to GitHub can be slow and may cause your repository to become bloated. To address this issue, you can use Git Large File Storage (Git LFS) to version large files and keep your repository size small.

#### If you are working with large files like .ubyte files in Git, you may run into issues with file size limits and slow upload/download times. Git LFS (Large File Storage) is a Git extension that allows you to store large files outside of your Git repository, while still being able to version and track those files.


* To use Git LFS for your .ubyte files, follow these steps:

* 1.Install Git LFS by following the instructions for your operating system here.
* 2.Once Git LFS is installed, navigate to your Git repository and run the following command to initialize Git LFS for your repository:

    ``` 
    git lfs install 
    git lfs track "*.ubyte"
   
    ```

* In the root directory of your repository, create a .gitattributes file and add the following line to it:
    ```
    git add .gitattributes
    git add .
    git commit -m "Add MNIST dataset"
    git push origin main 
    ```
 
 * Add your .ubyte files to your Git repository and commit the changes.
 
