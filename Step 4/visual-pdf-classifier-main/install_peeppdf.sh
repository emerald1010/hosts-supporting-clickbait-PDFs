### RUN THIS INSTALLER TO GET ALL THE PROJECT'S DEPENDENCIES ###

# Install PeepPDF
cd Data/FileTypes/PDF/Dependencies/PeepPDF
git clone https://github.com/jesparza/peepdf.git
cd peepdf
mv * ../
cd ..
rm -r peepdf
