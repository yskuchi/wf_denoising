void make_noisy_wf_macro(TString filename1, TString filename2)
{
   //% ./megbartender -I 'make_noisy_wf_macro.C("wf11100.root", "wf328469.root")'
   //% root 'make_noisy_wf_macro.C("wf11100.root", "wf328469.root")'

   TChain *wf1 = new TChain("wf");
   wf1->Add(filename1);
   auto nentry1 = wf1->GetEntries();
   cout<<"Number of events = "<<nentry1<<endl;

   TChain *wf2 = new TChain("wf");
   wf2->Add(filename2);
   auto nentry2 = wf2->GetEntries();
   cout<<"Number of events = "<<nentry2<<endl;


   TString rootfilename = filename1;
   rootfilename.ReplaceAll("wf", "wfnoisy");
   rootfilename = basename(rootfilename.Data());
   TFile rootout(rootfilename, "RECREATE", "wf");
   TTree *tree = new TTree("wf", "Tree for waveform");
   std::vector<Float_t> drsori(1024), drsnoisy(1024);
   tree->Branch("drsori", &drsori);
   tree->Branch("drsnoisy", &drsnoisy);   

   
   Int_t maxEvent = std::min(nentry1, nentry2);
   Int_t wfRange[] = {0, 1024};

   
   TTreeReader reader1(wf1);
   TTreeReaderArray<Float_t>   drs1(reader1, "drs");
   TTreeReader reader2(wf2);
   TTreeReaderArray<Float_t>   drs2(reader2, "drs");

   Int_t iEvent(-1);
   Int_t nTotData(0);
   while (reader1.Next() && reader2.Next()) {
      ++iEvent;
      if (maxEvent > 0 && iEvent > maxEvent) break;

      std::transform(drs1.begin(), drs1.end(), drs2.begin(), drsnoisy.begin(), std::plus<Float_t>());
      copy(drs1.begin(), drs1.end(), drsori.begin());
      tree->Fill();
      ++nTotData;
   }
   cout<<"Total number of samples: "<<nTotData<<endl;
   tree->Write();
   rootout.Close();
}
