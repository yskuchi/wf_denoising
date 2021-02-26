void read_wf_macro(TString filename)
{
   //% ./megbartender -I 'read_wf_macro.C("raw11100.root")'

   // Int_t addressSelect[] = {20480, 35319}; // MC
   // Int_t addressSelect[] = {2560, 2775}; // 2018
   Int_t addressSelect[] = {6176, 7671}; // 2020
   
   TChain *raw = new TChain("raw");
   raw->Add(filename);
   auto nentry = raw->GetEntries();
   cout<<"Number of events = "<<nentry<<endl;

   TString csvfile = filename;
   csvfile.ReplaceAll(".root", ".csv");
   csvfile.ReplaceAll("raw", "wf");
   csvfile = basename(csvfile.Data());
   ofstream fout(csvfile.Data(), std::ios::out);

   TString rootfilename = filename;
   rootfilename.ReplaceAll("raw", "wf");
   rootfilename = basename(rootfilename.Data());
   TFile rootout(rootfilename, "RECREATE", "wf");
   TTree *tree = new TTree("wf", "Tree for waveform");
   std::vector<Float_t> drs(1024);
   auto branch = tree->Branch("drs", &drs);
   //tree->Branch("drs", &drs[0], "drs[1024]/F");

   
   //Int_t maxEvent = 40;// 200;//-1; // MC
   Int_t maxEvent = 500;//-1;   // 2018
   Int_t wfRange[] = {0, 1024};

   
   TTreeReader reader(raw);
   TTreeReaderArray<MEGDRSChip>   chipRA(reader, "drschip");

   MEGDRSWaveform wf[8];
   Int_t iEvent(-1);
   Int_t nTotData(0);
   while (reader.Next()) {
      ++iEvent;
      if (maxEvent > 0 && iEvent > maxEvent) break;
      
      for (auto&& chip: chipRA) {
         auto chipData = chip.GetDRSChipData();
         auto address = chipData->GetAddress();
         if (address < addressSelect[0] || address > addressSelect[1]) continue;
         if (!chipData->HaveData()) continue;
         for (Int_t iChannel = 0; iChannel < 8; iChannel++) {
            chipData->SetWaveformAt(iChannel, &wf[iChannel]);
         }
         chipData->DecodeWaves();
         string line;
         for (Int_t iChannel = 0; iChannel < 8; iChannel++) {
            if (!wf[iChannel].GetNPoints()) continue;

            auto ampl = wf[iChannel].GetAmplitude();
            std::copy(ampl, ampl+1024, drs.begin());
            tree->Fill();
            
            for_each(ampl, ampl+1024, [&](Double_t a){line += std::to_string(a)+",";});
            line.pop_back();
            line += '\n';
            ++nTotData;
         }
         fout<<line;
      }
   }
   cout<<"Total number of samples: "<<nTotData<<endl;
   fout.close();
   tree->Write();
   rootout.Close();
}
