void read_wf_macro(TString filename)
{
   //% ./meganalyzer -I 'read_wf_macro.C("raw11100.root")'


   Int_t kNPoints = 512; // number of points of a wf to be output
   
   TChain *raw = new TChain("raw");
   raw->Add(filename);
   auto nentry = raw->GetEntries();
   cout<<"Number of events = "<<nentry<<endl;

   // rec file to get CYLDCHWireRunHeader
   TString recfilename = filename;
   recfilename.ReplaceAll("raw", "rec");
   TFile *recfile = TFile::Open(recfilename);
   auto pWireRunHeaders = (TClonesArray*)(recfile->Get("CYLDCHWireRunHeader"));

   vector<Int_t> addresses; // reorder addresses by wire IDs.
   for (Int_t iWire = 0; iWire < pWireRunHeaders->GetSize(); iWire++) {
      auto pWireRunHeader = static_cast<MEGCYLDCHWireRunHeader*>(pWireRunHeaders->At(iWire));
      if (!pWireRunHeader->GetActive()) continue;
      if (pWireRunHeader->GetDRSAddress_u() < 0 || pWireRunHeader->GetDRSAddress_d() < 0) continue;
      addresses.push_back(pWireRunHeader->GetDRSAddress_u());
      addresses.push_back(pWireRunHeader->GetDRSAddress_d());      
   }
   auto addressMin = std::min_element(addresses.begin(), addresses.end());
   auto addressMax = std::max_element(addresses.begin(), addresses.end());
   // Int_t addressSelect[] = {20480, 35319}; // MC
   // Int_t addressSelect[] = {2560, 2775}; // 2018
   // Int_t addressSelect[] = {6176, 7671}; // 2020
   Int_t addressSelect[] = {*addressMin, *addressMax}; // 2021 or new MC
   cout<<"Address "<<*addressMin<<" - "<<*addressMax<<endl;
   
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

   
   Int_t maxEvent = 50;// 200;//-1; // MC
   //Int_t maxEvent = 500;//-1;   // 2018
   
   TTreeReader reader(raw);
   TTreeReaderArray<MEGDRSChip>   chipRA(reader, "drschip");

   MEGDRSWaveform wf[8];
   Int_t iEvent(-1);
   Int_t nTotData(0);
   Int_t nChPrev(0);
   while (reader.Next()) {
      ++iEvent;
      if (maxEvent > 0 && iEvent > maxEvent) break;

      map<Int_t, vector<Float_t> > wfs;
      for (auto&& chip: chipRA) {
         auto chipData = chip.GetDRSChipData();
         auto address = chipData->GetAddress();
         if (address < addressSelect[0] || address > addressSelect[1]) continue;
         if (!chipData->HaveData()) continue;
         for (Int_t iChannel = 0; iChannel < 8; iChannel++) {
            chipData->SetWaveformAt(iChannel, &wf[iChannel]);
         }
         chipData->DecodeWaves();
         for (Int_t iChannel = 0; iChannel < 8; iChannel++) {
            if (!wf[iChannel].GetNPoints()) continue;
            Int_t addressCh = address + iChannel;
            auto ampl = wf[iChannel].GetAmplitude();
            // drs.assign(1024, 0);
            // std::copy(ampl, ampl+wf[iChannel].GetNPoints(), drs.begin());
            // wfs[addressCh] = drs;
            wfs[addressCh].resize(wf[iChannel].GetNPoints(), 0);
            std::copy(ampl, ampl+wf[iChannel].GetNPoints(), wfs[addressCh].begin());
         }
      }
      Int_t nCh(0);
      for (size_t iCh = 0; iCh < addresses.size(); iCh += 2) {
         if (wfs.find(addresses[iCh]) != wfs.end()
             && wfs.find(addresses[iCh + 1]) != wfs.end()) {
            string line;
            for (Int_t iEnd = 0; iEnd < 2; iEnd++) {
               drs.clear();
               drs.resize(1024, 0);
               std::copy(wfs[addresses[iCh + iEnd]].begin(),
                         wfs[addresses[iCh + iEnd]].end(), drs.begin());
               drs.resize(kNPoints);
               
               tree->Fill();
            
               for_each(drs.begin(), drs.end(), [&](Double_t a) {line += std::to_string(a)+",";});
               line.pop_back();
               line += '\n';
               ++nCh;
               ++nTotData;
            }
            fout<<line;
         }
      }
      if (nCh != nChPrev) {
         cout<<"Number of active channels "<<nCh<<endl;
         nChPrev = nCh;
      }
   }
   cout<<"Total number of samples: "<<nTotData<<endl;
   fout.close();
   tree->Write();
   rootout.Close();
}
