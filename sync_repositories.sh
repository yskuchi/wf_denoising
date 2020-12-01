git pull origin master
git pull bitbucket master
git push origin master
git push bitbucket master

cd ../wf_denoising_clean
git pull origin master

rsync -auhv --exclude=.git* --exclude sync_repositories.sh ./ $MEG2SYS/analyzer/macros/cyldch/WaveformAnalysisML/wf_denoising/
