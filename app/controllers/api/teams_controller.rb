class Api::TeamsController < ApplicationController
  def show
    @team = Team.includes(players: [:battings]).find(params[:id])
    render json: {team: @team, players: @team.players}.to_json
  end
end
